# README

## 秦菲Qinphy的小站

VSCode配置虚拟环境：

1. 安装插件 - Python Environments
2. 命令行新建虚拟环境：
    ```shell
    python -m venv .venv
    ```
3. 激活虚拟环境：
    ```shell
    & <Project-Name>/.venv/Scripts/Activate.ps1
    ```

安装环境记录：
```shell
pip install mkdocs
pip install mkdocs-material
pip install mkdocs-glightbox
pip install mkdocs-git-revision-date-localized-plugin
pip install mkdocs-git-authors-plugin
pip install mkdocs-statistics-plugin
```

本地运行项目：
```shell
mkdocs serve
```