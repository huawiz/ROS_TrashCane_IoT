@echo off
setlocal enabledelayedexpansion

for /f "delims=" %%i in ('dir 1_* /b /a-d') do (
    set "filename=%%i"
    set "newcontent="
    set /p oldcontent=< "%%i"
    set "firstchar=!oldcontent:~0,1!"
    set "restcontent=!oldcontent:~1!"
    set "newcontent=1!restcontent!"
    echo!newcontent! > "%%i"
)