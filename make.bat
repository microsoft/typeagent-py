:: Copyright (c) Microsoft Corporation.
:: Licensed under the MIT License.

@echo off
if "%~1"=="" goto help

if /I "%~1"=="help" goto help
if /I "%~1"=="venv" goto venv
if /I "%~1"=="check" goto check
if /I "%~1"=="test" goto test
if /I "%~1"=="demo" goto demo
if /I "%~1"=="clean" goto clean
if /I "%~1"=="format" goto format
if /I "%~1"=="build" goto build

echo Unknown command: %~1
goto help

:format
if not exist "venv\" call make.bat venv
echo Formatting code...
venv\Scripts\black typeagent *.py
goto end

:check
if not exist "venv\" call make.bat venv
echo Running checks...
venv\Scripts\pyright --pythonpath venv\Scripts\python typeagent *.py
goto end

:test
if not exist "venv\" call make.bat venv
echo Running tests...
venv\Scripts\python -m typeagent.podcasts testdata\npr.txt
goto end

:demo
if not exist "venv\" call make.bat venv
echo Running demo...
venv\Scripts\python demo.py
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
venv\Scripts\pyright --version
venv\Scripts\black --version

goto end

:clean
echo Sorry, you have to clean up manually.
echo These are to be deleted: build dist *.egg-info venv
echo Delete venv if the requirements have changed.
echo The others are products of the build step.
goto end

:help
echo Usage: make.bat [format^|check^|test^|demo^|build^|venv^|clean^|help]
goto end

:end
