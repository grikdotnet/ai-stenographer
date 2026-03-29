param(
    [Parameter(Mandatory)]
    [string]$ServerUrl,

    [string]$InputFile
)

$env:STT_SERVER_URL = $ServerUrl
if ($InputFile) {
    $env:STT_INPUT_FILE = $InputFile
}

npm run tauri:dev
