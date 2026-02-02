@echo off
chcp 65001 >nul
echo 正在处理路径: %*
echo 授予完全控制权限（为删除nul文件做准备）...

:: 获取管理员权限并为目标路径设置完全控制（保留原权限逻辑）
@echo Y|cacls %* /t /e /c /g Everyone:f >nul 2>&1

echo 正在查找并删除所有nul文件...
:: 遍历目标路径（含子目录），仅删除名为nul的文件，支持长路径/特殊名
for /r "%~1" %%i in (nul) do (
    if exist "%%i" (
        echo 正在删除: "%%i"
        :: 长路径支持 + 强制删除文件（仅删文件，不碰文件夹）
        DEL /F /A /Q \\?\"%%i" >nul 2>&1
    )
)

echo nul文件删除完成！
pause