// Prevents additional console window on Windows in release, DO NOT REMOVE!!
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

fn main() {
    if llm_wiki_lib::omnimind_server::should_start_from_args(std::env::args()) {
        return llm_wiki_lib::omnimind_server::start();
    }

    llm_wiki_lib::run();
}
