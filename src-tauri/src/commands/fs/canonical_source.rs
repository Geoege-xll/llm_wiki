//! OmniMind 的规范源证据边界：复用原生提取并对完整字节签名，不把模型生成文字当成原件。
//! 来源哈希与精确文件清单共同授权；索引中保留了路径并不意味着当前字节仍受信任。
use calamine::{open_workbook_auto, Data, Reader};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::Path;

pub const EXTRACTOR_VERSION: &str = "omnimind-canonical-source-v1";
pub const INVALID_HASHES: &str = "INVALID_SOURCE_HASHES";
pub const CONTENT_CHANGED: &str = "SOURCE_CONTENT_CHANGED";

pub fn sha256(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

/// read_file 为桌面预览保留了“无法提取”的占位字符串；这些不是事实来源，不能被签发。
/// 纯文本允许包含相同字样，只有结构化文档提取器的已知占位输出才按失败处理。
pub fn validate_text(content: &str, extension: &str) -> Result<(), &'static str> {
    let text = content.trim();
    if text.is_empty() {
        return Err("EMPTY_CANONICAL_SOURCE");
    }
    if matches!(
        extension,
        "doc" | "docx" | "pptx" | "ppt" | "odt" | "odp" | "pdf" | "xls" | "xlsx" | "ods"
    ) && ((text.starts_with("[Could not extract") && text.ends_with(']'))
        || (text.starts_with("[Document:") && text.ends_with(']')))
    {
        return Err("CANONICAL_EXTRACTION_UNAVAILABLE");
    }
    Ok(())
}

/// 沿用 Core 的 calamine → Markdown 提取规则，逐个读取全部工作表。
/// f64::to_string 使用最短可往返表示，禁止展示层的两位小数格式化篡改 0.001 等事实。
/// 任一非空工作表读取出错时整份规范源失败，避免首表成功掩盖后续资料缺失。
pub fn extract_spreadsheet(path: &str) -> Result<String, String> {
    let mut workbook =
        open_workbook_auto(path).map_err(|_| "Unable to open canonical spreadsheet".to_string())?;
    let sheet_names = workbook.sheet_names().to_vec();
    let mut result = String::new();
    for sheet_name in &sheet_names {
        let range = workbook
            .worksheet_range(sheet_name)
            .map_err(|_| "Unable to extract every spreadsheet sheet".to_string())?;
        if range.is_empty() {
            continue;
        }
        let rows: Vec<Vec<String>> = range
            .rows()
            .map(|row| {
                row.iter()
                    .map(|cell| match cell {
                        Data::Empty => String::new(),
                        Data::String(value) => value.clone(),
                        Data::Float(value) => value.to_string(),
                        Data::Int(value) => value.to_string(),
                        Data::Bool(value) => value.to_string(),
                        Data::DateTime(value) => value.to_string(),
                        Data::DateTimeIso(value) | Data::DurationIso(value) => value.clone(),
                        Data::Error(value) => format!("ERR:{value:?}"),
                    })
                    .collect()
            })
            .collect();
        if !rows.iter().flatten().any(|cell| !cell.trim().is_empty()) {
            continue;
        }
        if sheet_names.len() > 1 {
            result.push_str(&format!("## {sheet_name}\n\n"));
        }
        let max_cols = rows.iter().map(Vec::len).max().unwrap_or(0);
        for (index, row) in rows.iter().enumerate() {
            let mut padded = row.clone();
            padded.resize(max_cols, String::new());
            let escaped: Vec<String> = padded.iter().map(|cell| cell.replace('|', "\\|")).collect();
            result.push_str("| ");
            result.push_str(&escaped.join(" | "));
            result.push_str(" |\n");
            if index == 0 {
                result.push('|');
                for _ in 0..max_cols {
                    result.push_str(" --- |");
                }
                result.push('\n');
            }
        }
        result.push('\n');
    }
    validate_text(&result, "xlsx").map_err(str::to_string)?;
    Ok(result)
}

/// 固定请求中的来源版本。正文被扫描前后都核验；候选正文另行核验，关闭 ABA 变化窗口。
/// 外层先执行既有路径/租户校验；本层只接受已经规范化的精确项目相对路径集合。
pub struct SourceHashScope {
    expected: BTreeMap<String, String>,
}

impl SourceHashScope {
    pub fn new(
        allowed: Option<&BTreeSet<String>>,
        hashes: Option<&BTreeMap<String, String>>,
    ) -> Result<Option<Self>, &'static str> {
        let Some(hashes) = hashes else {
            return Ok(None);
        };
        let Some(allowed) = allowed else {
            return Err(INVALID_HASHES);
        };
        if hashes.keys().collect::<BTreeSet<_>>() != allowed.iter().collect::<BTreeSet<_>>()
            || hashes.values().any(|value| {
                value.len() != 64
                    || !value
                        .bytes()
                        .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
            })
        {
            return Err(INVALID_HASHES);
        }
        Ok(Some(Self {
            expected: hashes.clone(),
        }))
    }

    pub fn verify_files(&self, project: &str) -> Result<(), &'static str> {
        for (relative, expected) in &self.expected {
            // 每次重新检查路径组件，拒绝检索过程中被换成链接的源文件或目录。
            let mut target = Path::new(project).to_path_buf();
            for component in relative.split('/') {
                target.push(component);
                if fs::symlink_metadata(&target)
                    .map_err(|_| CONTENT_CHANGED)?
                    .file_type()
                    .is_symlink()
                {
                    return Err(CONTENT_CHANGED);
                }
            }
            let bytes = fs::read(&target).map_err(|_| CONTENT_CHANGED)?;
            if sha256(&bytes) != *expected {
                return Err(CONTENT_CHANGED);
            }
        }
        Ok(())
    }

    pub fn verify_candidate(&self, path: &str, content: Option<&str>) -> Result<(), &'static str> {
        let expected = self.expected.get(path).ok_or(CONTENT_CHANGED)?;
        if content.map(|text| sha256(text.as_bytes())).as_ref() != Some(expected) {
            return Err(CONTENT_CHANGED);
        }
        Ok(())
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use std::io::{Cursor, Write};

    /// 真实最小 OOXML 文件，第二张表单独持有关键小数；不以解析器 Mock 伪造多表成功。
    pub(crate) fn xlsx_fixture(second_sheet: Option<&str>) -> Vec<u8> {
        let mut archive = zip::ZipWriter::new(Cursor::new(Vec::new()));
        let options = zip::write::SimpleFileOptions::default();
        let entries = [
            (
                "[Content_Types].xml",
                r#"<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types"><Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/><Default Extension="xml" ContentType="application/xml"/><Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/><Override PartName="/xl/worksheets/sheet1.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/><Override PartName="/xl/worksheets/sheet2.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/></Types>"#,
            ),
            (
                "_rels/.rels",
                r#"<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/></Relationships>"#,
            ),
            (
                "xl/workbook.xml",
                r#"<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"><sheets><sheet name="基础" sheetId="1" r:id="rId1"/><sheet name="精密参数" sheetId="2" r:id="rId2"/></sheets></workbook>"#,
            ),
            (
                "xl/_rels/workbook.xml.rels",
                r#"<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet1.xml"/><Relationship Id="rId2" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet2.xml"/></Relationships>"#,
            ),
            (
                "xl/worksheets/sheet1.xml",
                r#"<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"><sheetData><row r="1"><c r="A1" t="inlineStr"><is><t>净重</t></is></c><c r="B1"><v>4.3</v></c></row></sheetData></worksheet>"#,
            ),
        ];
        for (name, content) in entries {
            archive.start_file(name, options).unwrap();
            archive.write_all(content.as_bytes()).unwrap();
        }
        if let Some(content) = second_sheet {
            archive
                .start_file("xl/worksheets/sheet2.xml", options)
                .unwrap();
            archive.write_all(content.as_bytes()).unwrap();
        }
        archive.finish().unwrap().into_inner()
    }

    pub(crate) const PRECISE_SHEET: &str = r#"<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"><sheetData><row r="1"><c r="A1" t="inlineStr"><is><t>第二表独有误差</t></is></c><c r="B1"><v>0.001</v></c><c r="C1"><v>-0.000123456789</v></c><c r="D1"><v>123456789012.345</v></c></row></sheetData></worksheet>"#;

    #[tokio::test]
    async fn canonical_xlsx_preserves_every_sheet_and_numeric_precision_via_read_file() {
        let path = std::env::temp_dir().join(format!("canonical-{}.xlsx", uuid::Uuid::new_v4()));
        fs::write(&path, xlsx_fixture(Some(PRECISE_SHEET))).unwrap();
        let content = crate::commands::fs::read_file(path.to_string_lossy().into(), Some(false))
            .await
            .unwrap();
        for expected in [
            "## 基础",
            "净重",
            "4.3",
            "## 精密参数",
            "第二表独有误差",
            "0.001",
            "-0.000123456789",
            "123456789012.345",
        ] {
            assert!(content.contains(expected), "缺少完整规范事实: {expected}");
        }
        fs::remove_file(path).unwrap();
    }

    #[test]
    fn canonical_xlsx_missing_sheet_does_not_return_partial_success() {
        let path =
            std::env::temp_dir().join(format!("canonical-broken-{}.xlsx", uuid::Uuid::new_v4()));
        fs::write(&path, xlsx_fixture(None)).unwrap();
        assert!(extract_spreadsheet(path.to_str().unwrap()).is_err());
        fs::remove_file(path).unwrap();
    }

    #[test]
    fn canonical_rejects_empty_and_document_placeholders() {
        assert!(validate_text(" \n\t", "txt").is_err());
        assert!(validate_text("[Could not extract text from PPTX]", "pptx").is_err());
        assert!(validate_text(
            "[Document: file — text extraction not supported for .ppt format]",
            "ppt"
        )
        .is_err());
        assert!(validate_text("[Could not extract text from PPTX]", "txt").is_ok());
    }

    #[test]
    fn source_hash_scope_requires_exact_paths_and_lowercase_sha256() {
        let paths = BTreeSet::from(["wiki/a.md".into()]);
        let hashes = BTreeMap::from([("wiki/a.md".into(), sha256(b"correct"))]);
        assert!(SourceHashScope::new(Some(&paths), Some(&hashes))
            .unwrap()
            .is_some());
        assert!(SourceHashScope::new(Some(&paths), None).unwrap().is_none());
        assert_eq!(
            SourceHashScope::new(None, Some(&hashes)).err(),
            Some(INVALID_HASHES)
        );
        let mut wrong = hashes.clone();
        wrong.insert("wiki/b.md".into(), sha256(b"extra"));
        assert_eq!(
            SourceHashScope::new(Some(&paths), Some(&wrong)).err(),
            Some(INVALID_HASHES)
        );
        wrong.remove("wiki/b.md");
        wrong.insert("wiki/a.md".into(), sha256(b"correct").to_uppercase());
        assert_eq!(
            SourceHashScope::new(Some(&paths), Some(&wrong)).err(),
            Some(INVALID_HASHES)
        );
    }

    #[test]
    fn source_hash_scope_detects_rewrites_deletion_and_aba_candidate() {
        let root = std::env::temp_dir().join(format!("canonical-scope-{}", uuid::Uuid::new_v4()));
        fs::create_dir_all(root.join("wiki")).unwrap();
        let path = root.join("wiki/a.md");
        fs::write(&path, "correct").unwrap();
        let paths = BTreeSet::from(["wiki/a.md".into()]);
        let hashes = BTreeMap::from([("wiki/a.md".into(), sha256(b"correct"))]);
        let scope = SourceHashScope::new(Some(&paths), Some(&hashes))
            .unwrap()
            .unwrap();
        scope.verify_files(root.to_str().unwrap()).unwrap();
        fs::write(&path, "changed").unwrap();
        assert_eq!(
            scope.verify_files(root.to_str().unwrap()),
            Err(CONTENT_CHANGED)
        );
        fs::write(&path, "correct").unwrap();
        scope.verify_files(root.to_str().unwrap()).unwrap();
        assert_eq!(
            scope.verify_candidate("wiki/a.md", Some("changed")),
            Err(CONTENT_CHANGED)
        );
        assert_eq!(
            scope.verify_candidate("wiki/b.md", Some("correct")),
            Err(CONTENT_CHANGED)
        );
        scope
            .verify_candidate("wiki/a.md", Some("correct"))
            .unwrap();
        fs::remove_file(path).unwrap();
        assert_eq!(
            scope.verify_files(root.to_str().unwrap()),
            Err(CONTENT_CHANGED)
        );
        fs::remove_dir_all(root).unwrap();
    }
}
