# Creates the capstone topics on the running broker (idempotent).
$ErrorActionPreference = 'Stop'

$repoRoot = (Resolve-Path "$PSScriptRoot\..\..").Path
$java = Get-ChildItem "$repoRoot\day01_mini_lakehouse\.runtime" -Directory -Filter 'jdk-17*' | Select-Object -First 1
$env:JAVA_HOME = $java.FullName
$env:PATH = "$env:JAVA_HOME\bin;$env:PATH"

$kafka = if ($env:KAFKA_HOME) { $env:KAFKA_HOME } else { 'C:\kfk\kafka' }
$bootstrap = if ($env:KAFKA_BOOTSTRAP_SERVERS) { $env:KAFKA_BOOTSTRAP_SERVERS } else { 'localhost:9092' }

foreach ($t in @('capstone.readings', 'capstone.dlq')) {
    Write-Host "Creating topic $t ..."
    & "$kafka\bin\windows\kafka-topics.bat" --bootstrap-server $bootstrap `
        --create --if-not-exists --topic $t --partitions 1 --replication-factor 1
}

Write-Host "`nTopics on ${bootstrap}:"
& "$kafka\bin\windows\kafka-topics.bat" --bootstrap-server $bootstrap --list
