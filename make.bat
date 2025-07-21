:: Copyright (c) Microsoft Corporation.
:: Licensed under the MIT License.

:: This is a batch file to run common actions.
:: It can format the code, check the code, run the tests,
:: build the package, create a virtual environment, and clean up.
:: To avoid having to type `./make` all the time,
:: use `set-alias make ".\make.bat"` in PowerShell.

@echo off
if "%~1"=="" goto help

if /I "%~1"=="format" goto format
if /I "%~1"=="check" goto check
if /I "%~1"=="test" goto test
if /I "%~1"=="demo" goto demo
if /I "%~1"=="build" goto build
if /I "%~1"=="venv" goto venv
if /I "%~1"=="install-uv" goto install-uv
if /I "%~1"=="clean" goto clean
if /I "%~1"=="help" goto help

echo Unknown command: %~1
goto help

:format
if not exist ".venv\" call make.bat venv
echo Formatting code...
.venv\Scripts\black typeagent test
goto end

:check
if not exist ".venv\" call make.bat venv
echo Running checks...
.venv\Scripts\pyright --pythonpath .venv\Scripts\python typeagent test
goto end

:test
if not exist ".venv\" call make.bat venv
echo Running tests...
.venv\Scripts\python -m pytest test
goto end

:demo
if not exist ".venv\" call make.bat venv
echo Running demo...
.venv\Scripts\python -m tools.utool
goto end

:build
if not exist ".venv\" call make.bat venv
echo Building package...
.venv\Scripts\python -m build --wheel --installer uv
goto end

:venv
echo Creating virtual environment...
uv sync -q
.venv\Scripts\python --version
.venv\Scripts\black --version
.venv\Scripts\pyright --version
.venv\Scripts\python -m pytest --version
goto end

:install-uv
echo Installing uv requires Administrator mode!
echo 1. Using PowerShell in Administrator mode:
echo    Invoke-RestMethod https://astral.sh/uv/install.ps1 ^| Invoke-Expression
echo 2. Add ~/.local/bin to $env:PATH, e.g. by putting
echo        $env:PATH += ";$HOME\.local\bin
echo    in your PowerShell profile ($PROFILE) and restarting PowerShell.
echo    (Sorry, I have no idea how to do that in cmd.exe.)
goto end

:clean
if exist build rmdir /s /q build
if exist dist rmdir /s /q dist
if exist typeagent.egg-info rmdir /s /q typeagent.egg-info
if exist .venv rmdir /s /q .venv
if exist .pytest_cache rmdir /s /q .pytest_cache
goto end

:help
echo Usage: .\make [format^|check^|test^|build^|venv^|install-uv^|clean^|help]
goto end

:end
