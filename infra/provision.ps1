<#
Provisions the Azure resources for the fault-tolerant agent project INSIDE an existing resource group
(it never creates or deletes resource groups, so it works with restricted roles):
  - one Azure AI Foundry resource (kind AIServices, SKU S0)
  - two minimal-capacity deployments: "primary" and "fallback"
Writes endpoint + deployment names to ../.env (no keys; auth is Entra ID).

Cost-conscious by design: deployments are pay-per-token, capacity is the minimum (1K TPM), nothing is always-on.

Usage:  ./infra/provision.ps1 -ResourceGroup <name> [-Prefix ftagent] [-WhatIf]
        (-WhatIf prints the plan and creates nothing)
Prereq: az login; az account set --subscription <id>
#>
param(
  [Parameter(Mandatory = $true)][string]$ResourceGroup,   # an existing resource group you can write to
  [string]$Location        = "westus",
  [string]$Prefix          = "ftagent",
  [string]$PrimaryModel    = "gpt-4.1-nano",  [string]$PrimaryVersion  = "2025-04-14", [string]$PrimarySku  = "GlobalStandard",
  [string]$FallbackModel   = "gpt-4.1-mini",  [string]$FallbackVersion = "2025-04-14", [string]$FallbackSku = "Standard",
  [int]   $Capacity        = 1,
  [switch]$WhatIf
)
$ErrorActionPreference = "Stop"
$az = if (Get-Command az -ErrorAction SilentlyContinue) { "az" } else { "C:\Program Files\Microsoft SDKs\Azure\CLI2\wbin\az.cmd" }

& $az account show --output none 2>$null
if ($LASTEXITCODE -ne 0) { throw "Not logged in. Run: az login --use-device-code" }

$sub  = & $az account show --query id -o tsv
$name = "$Prefix-foundry-$($sub.Substring(0,6))"   # globally unique custom subdomain

Write-Host "Plan: account '$name' in RG '$ResourceGroup' ($Location)"
Write-Host "  primary  = $PrimaryModel $PrimaryVersion ($PrimarySku, ${Capacity}K TPM)"
Write-Host "  fallback = $FallbackModel $FallbackVersion ($FallbackSku, ${Capacity}K TPM)"
if ($WhatIf) { Write-Host "WhatIf: nothing created."; exit 0 }

& $az cognitiveservices account create `
  --name $name --resource-group $ResourceGroup --location $Location `
  --kind AIServices --sku S0 --custom-domain $name --tags project=fault-tolerant-agent --yes --output none

function New-Deployment($dep, $model, $version, $sku) {
  & $az cognitiveservices account deployment create `
    --name $name --resource-group $ResourceGroup --deployment-name $dep `
    --model-name $model --model-version $version --model-format OpenAI `
    --sku-name $sku --sku-capacity $Capacity --output none
  if ($LASTEXITCODE -ne 0) { throw "Deployment $dep ($model) failed" }
}
New-Deployment "primary"  $PrimaryModel  $PrimaryVersion  $PrimarySku
New-Deployment "fallback" $FallbackModel $FallbackVersion $FallbackSku

$endpoint = & $az cognitiveservices account show --name $name --resource-group $ResourceGroup --query properties.endpoint -o tsv
@"
AZURE_AI_ENDPOINT=$endpoint
PRIMARY_MODEL=primary
FALLBACK_MODEL=fallback
AZURE_RESOURCE_GROUP=$ResourceGroup
AZURE_ACCOUNT_NAME=$name
"@ | Set-Content -Encoding utf8 (Join-Path $PSScriptRoot "..\.env")

Write-Host "Done. Wrote .env. Tear down with: ./infra/teardown.ps1"
