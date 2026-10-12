//! OmniMind 全量索引的持久写入栅栏。
//!
//! 此模块不持有全局状态：每次均从项目私有目录读取最高代次，因此 Core 重启不会
//! 重新接受已经失效的请求。调用者必须在同一个项目写锁内完成 reserve，或完成
//! check_write 与 LanceDB 提交；不能先检查再释放锁，否则旧 HTTP 仍可迟到覆盖。
use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::Path;

pub const CONFLICT: &str = "VECTOR_FENCE_CONFLICT";
pub const INVALID: &str = "INVALID_VECTOR_FENCE";
const STORAGE_ERROR: &str = "VECTOR_FENCE_STORAGE_FAILED";
const FILE_NAME: &str = "vector-build-fence";

/// wire 使用规范十进制字符串，避免 JSON 数字经过 JS 时损失 u64 精度。
/// 不接受正号、前导零或空白，成功响应可原样返回调用方的 fence。
pub fn parse(value: &str) -> Result<u64, String> {
    if value.is_empty()
        || !value.bytes().all(|byte| byte.is_ascii_digit())
        || (value.len() > 1 && value.starts_with('0'))
    {
        return Err(INVALID.into());
    }
    value.parse::<u64>().map_err(|_| INVALID.into())
}

fn read(project: &str) -> Result<Option<u64>, String> {
    let path = Path::new(project).join(".llm-wiki").join(FILE_NAME);
    match fs::read_to_string(path) {
        Ok(value) => parse(&value).map(Some).map_err(|_| STORAGE_ERROR.into()),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(None),
        // 损坏或权限故障不能当作“从未设置”，否则会重新开放旧协议写入。
        Err(_) => Err(STORAGE_ERROR.into()),
    }
}

pub fn check_write(project: &str, supplied: Option<&str>) -> Result<(), String> {
    let requested = supplied.map(parse).transpose()?;
    match (read(project)?, requested) {
        (None, None) => Ok(()), // 尚未采用 fence 的原生项目保留旧合同。
        (Some(current), Some(value)) if current == value => Ok(()),
        _ => Err(CONFLICT.into()), // 新协议必须先 reserve；不能用 replace 偷偷取得代次。
    }
}

/// 原子替换并同步目录元数据；只有最高代次已经持久化才允许返回 reserve 成功。
/// 私有文件不位于 Lance 表目录，清理或重建表也不会删除保护状态。
pub fn reserve(project: &str, supplied: &str) -> Result<(), String> {
    let requested = parse(supplied)?;
    if let Some(current) = read(project)? {
        if requested < current {
            return Err(CONFLICT.into());
        }
        if requested == current {
            return Ok(());
        }
    }
    let directory = Path::new(project).join(".llm-wiki");
    fs::create_dir_all(&directory).map_err(|_| STORAGE_ERROR.to_string())?;
    let temporary = directory.join(format!(".vector-build-fence-{}", uuid::Uuid::new_v4()));
    let result = (|| -> std::io::Result<()> {
        let mut options = OpenOptions::new();
        options.write(true).create_new(true);
        #[cfg(unix)]
        {
            use std::os::unix::fs::OpenOptionsExt;
            options.mode(0o600);
        }
        let mut file = options.open(&temporary)?;
        file.write_all(supplied.as_bytes())?;
        file.sync_all()?;
        fs::rename(&temporary, directory.join(FILE_NAME))?;
        // Unix 上目录 fsync 保证 rename 的目录项在成功响应前已经落盘。
        #[cfg(unix)]
        fs::File::open(&directory)?.sync_all()?;
        Ok(())
    })();
    if result.is_err() {
        let _ = fs::remove_file(temporary);
    }
    result.map_err(|_| STORAGE_ERROR.into())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fence_is_strict_unsigned_decimal() {
        for invalid in ["", " 1", "+1", "-1", "01", "1.0", "18446744073709551616"] {
            assert_eq!(parse(invalid).unwrap_err(), INVALID);
        }
        assert_eq!(parse("18446744073709551615").unwrap(), u64::MAX);
    }

    #[test]
    fn fence_reopens_persisted_highest_and_fails_closed_on_corruption() {
        let directory = std::env::temp_dir().join(format!("wiki-fence-{}", uuid::Uuid::new_v4()));
        let project = directory.to_str().unwrap();
        assert!(check_write(project, None).is_ok());
        assert_eq!(check_write(project, Some("10")).unwrap_err(), CONFLICT);
        reserve(project, "10").unwrap();
        reserve(project, "10").unwrap();
        reserve(project, "11").unwrap();
        assert_eq!(reserve(project, "10").unwrap_err(), CONFLICT);
        // 没有内存缓存：新读者只靠磁盘最高代次，也会拒绝旧请求与无 fence 请求。
        assert!(check_write(project, Some("11")).is_ok());
        assert_eq!(check_write(project, Some("10")).unwrap_err(), CONFLICT);
        assert_eq!(check_write(project, None).unwrap_err(), CONFLICT);
        fs::write(directory.join(".llm-wiki").join(FILE_NAME), "broken").unwrap();
        assert_eq!(check_write(project, None).unwrap_err(), STORAGE_ERROR);
        assert_eq!(reserve(project, "12").unwrap_err(), STORAGE_ERROR);
        fs::remove_dir_all(directory).unwrap();
    }
}
