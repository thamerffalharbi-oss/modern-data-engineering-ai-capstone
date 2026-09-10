# Starts a real single-node Apache Kafka broker in KRaft mode (no ZooKeeper).
# Windows-native: uses the bundled JDK 17 from day01_mini_lakehouse/.runtime.
#
# NOTE ON KAFKA_HOME: Kafka's Windows .bat launchers build one long -classpath
# argument. Under a deep path (Desktop\Engdata_ai\capstone\infra\...) that
# argument exceeds the 8191-character cmd.exe limit and the broker fails with
# "The input line is too long". Kafka is therefore installed at a short path.
# Override with $env:KAFKA_HOME if you unpack it elsewhere.
$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot

$repoRoot = (Resolve-Path "$PSScriptRoot\..\..").Path
$java = Get-ChildItem "$repoRoot\day01_mini_lakehouse\.runtime" -Directory -Filter 'jdk-17*' | Select-Object -First 1
if (-not $java) { throw 'JDK 17 not found under day01_mini_lakehouse\.runtime' }
$env:JAVA_HOME = $java.FullName
$env:PATH = "$env:JAVA_HOME\bin;$env:PATH"

$kafka = if ($env:KAFKA_HOME) { $env:KAFKA_HOME } else { 'C:\kfk\kafka' }
if (-not (Test-Path "$kafka\bin\windows\kafka-server-start.bat")) {
    throw "Kafka not found at $kafka. Run infra\kafka-download.ps1 first."
}

# Keep broker data next to the install so the classpath stays short.
$dataDir = 'C:\kfk\kafka-data'
$cfg = Join-Path $PSScriptRoot 'kraft-server.properties'

# Format the storage directory once (idempotent: skip if already formatted).
if (-not (Test-Path (Join-Path $dataDir 'meta.properties'))) {
    New-Item -ItemType Directory -Force $dataDir | Out-Null
    $uuid = (& "$kafka\bin\windows\kafka-storage.bat" random-uuid | Select-Object -Last 1).Trim()
    Write-Host "Formatting KRaft storage with cluster id $uuid"
    & "$kafka\bin\windows\kafka-storage.bat" format -t $uuid -c $cfg
}

Write-Host "Starting Kafka broker on localhost:9092 (KRaft mode)..."
& "$kafka\bin\windows\kafka-server-start.bat" $cfg
