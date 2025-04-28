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
if /I "%~1"=="build" goto build
if /I "%~1"=="venv" goto venv
if /I "%~1"=="clean" goto clean
if /I "%~1"=="help" goto help

echo Unknown command: %~1
goto help

:format
if not exist "venv\" call make.bat venv
echo Formatting code...
venv\Scripts\black typeagent
goto end

:check
if not exist "venv\" call make.bat venv
echo Running checks...
venv\Scripts\pyright --pythonpath venv\Scripts\python typeagent *.py
goto end

:test
if not exist "venv\" call make.bat venv
echo Running tests...
venv\Scripts\python -m pytest
goto end

:build
if not exist "venv\" call make.bat venv
echo Building package...
venv\Scripts\python -m build --wheel
goto end

:venv
echo Creating virtual environment...
python3.12 -m venv venv
venv\Scripts\pip -q install -r requirements.txt
venv\Scripts\python --version
venv\Scripts\black --version
venv\Scripts\pyright --version
venv\Scripts\python -m pytest --version

goto end

:clean
echo Sorry, you have to clean up manually.
echo These are to be deleted: build dist *.egg-info venv
echo Delete venv if the requirements have changed.
echo The others are products of the build step.
goto end

:help
echo Usage: .\make [format^|check^|test^|build^|venv^|clean^|help]
goto end

:end
