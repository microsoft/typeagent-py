:: Copyright (c) Microsoft Corporation.
:: Licensed under the MIT License.

:: This is a batch file to run common actions.
:: It can format the code, check the code, run the tests,
:: build the package, create a virtual environment, and clean up.
:: To avoid having to type `./make` all the time,
:: use `set-alias make ".\make.bat"` in PowerShell.

@echo off
setlocal
if "%~1"=="" goto help

:: Remember the command, then collect every remaining argument into ARGS.
:: (The all-args variable can't be used here: 'shift' doesn't affect it.)
set "CMD=%~1"
set "ARGS="
:collect
shift
if "%~1"=="" goto collected
set "ARGS=%ARGS% %1"
goto collect

:collected
:: Drop the leading space left by the loop above.
if defined ARGS set "ARGS=%ARGS:~1%"

:dispatch
if /I "%CMD%"=="format" goto format
if /I "%CMD%"=="check" goto check
if /I "%CMD%"=="test" goto test
if /I "%CMD%"=="coverage" goto coverage
if /I "%CMD%"=="demo" goto demo
if /I "%CMD%"=="build" goto build
if /I "%CMD%"=="venv" goto venv
if /I "%CMD%"=="sync" goto sync
if /I "%CMD%"=="install-uv" goto install-uv
if /I "%CMD%"=="clean" goto clean
if /I "%CMD%"=="help" goto help

echo Unknown command: %CMD%
goto help

:: Extra arguments are passed on to the tools, e.g. '.\make format --check --diff'.
:format
if not exist ".venv\" call make.bat venv
echo Formatting code...
uv run isort src tests tools examples %ARGS% || exit /b 1
uv run black -tpy312 src tests tools examples %ARGS% || exit /b 1
goto end

:: intentionally running pyright only for the lowest and the highest version 
:: running it for all versions takes too much time and doesn't add enough diagnostic power
:: Keep the checked versions in sync with the 'check' target in the Makefile.
:check
if not exist ".venv\" call make.bat venv
echo Running type checks...
uv run pyright --pythonversion 3.12 src tests tools examples || exit /b 1
uv run pyright --pythonversion 3.15 src tests tools examples || exit /b 1
goto end

:test
if not exist ".venv\" call make.bat venv
echo Running unit tests...
uv run pytest %ARGS%
goto end

:coverage
setlocal
if not exist ".venv\" call make.bat venv
echo Running test coverage...
uv run coverage erase
set "COVERAGE_PROCESS_START=.coveragerc"
uv run coverage run -m pytest %ARGS%
uv run coverage combine
uv run coverage report
endlocal
goto end


:demo
if not exist ".venv\" call make.bat venv
echo Running query tool...
uv run python -m tools.query %ARGS%
goto end

:build
if not exist ".venv\" call make.bat venv
echo Building package...
uv build
goto end

:venv
echo Creating virtual environment...
uv sync -q
uv run python --version
uv run black --version
uv run pyright --version
uv run pytest --version
goto end

:sync
uv sync %ARGS%
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
echo Cleaning out build and dev artifacts...
if exist build rmdir /s /q build
if exist dist rmdir /s /q dist
if exist typeagent.egg-info rmdir /s /q typeagent.egg-info
if exist .venv rmdir /s /q .venv
if exist .pytest_cache rmdir /s /q .pytest_cache
goto end

:help
echo Usage: .\make [format^|check^|test^|coverage^|demo^|build^|venv^|sync^|install-uv^|clean^|help]
goto end

:end
