<#
Removes what provision.ps1 created: the Foundry account (its deployments go with it) and purges
the soft-deleted account so the name and quota are released. The resource group itself is left
alone. A role without purge rights leaves a soft-deleted (unbilled) record behind.
Usage: ./infra/teardown.ps1 [-Yes]
#>
param([switch]$Yes)
$ErrorActionPreference = "Stop"
$az = if (Get-Command az -ErrorAction SilentlyContinue) { "az" } else { "C:\Program Files\Microsoft SDKs\Azure\CLI2\wbin\az.cmd" }

$envFile = Join-Path $PSScriptRoot "..\.env"
if (-not (Test-Path $envFile)) { throw ".env not found; nothing recorded to delete." }
$cfg = @{}
Get-Content $envFile | ForEach-Object { if ($_ -match '^([^=]+)=(.*)$') { $cfg[$Matches[1]] = $Matches[2] } }
$rg = $cfg["AZURE_RESOURCE_GROUP"]; $name = $cfg["AZURE_ACCOUNT_NAME"]

# Safety: only delete an account tagged by this project.
$tag = & $az cognitiveservices account show --name $name --resource-group $rg --query "tags.project" -o tsv 2>$null
if ($tag -ne "fault-tolerant-agent") { Write-Host "Account '$name' not found or not tagged for this project. Nothing deleted."; exit 0 }
$loc = & $az cognitiveservices account show --name $name --resource-group $rg --query location -o tsv

if (-not $Yes) {
  if ((Read-Host "Delete account '$name' in '$rg' and purge it? (yes/no)") -ne "yes") { Write-Host "Aborted."; exit 1 }
}
& $az cognitiveservices account delete --name $name --resource-group $rg
if ($LASTEXITCODE -ne 0) { throw "Account delete failed; .env kept so you can retry." }
$ErrorActionPreference = "Continue"   # Windows PowerShell 5.1 turns native stderr into a terminating error under "Stop"
& $az cognitiveservices account purge --name $name --resource-group $rg --location $loc 2>$null
if ($LASTEXITCODE -ne 0) {
  Write-Warning "Purge not permitted for this role. The account is deleted (soft-deleted accounts are not billed); re-provisioning with the same name may need a different -Prefix."
}
Remove-Item $envFile
Write-Host "Teardown finished."
