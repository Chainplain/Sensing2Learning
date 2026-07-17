param(
    [string]$InputPath = "MULTIROCKET_TRAINING_REPORT.md",
    [string]$OutputPath = "tmp/pdfs/MULTIROCKET_TRAINING_REPORT.html"
)

$ErrorActionPreference = "Stop"
$root = (Get-Location).Path
$inputFull = [IO.Path]::GetFullPath((Join-Path $root $InputPath))
$outputFull = [IO.Path]::GetFullPath((Join-Path $root $OutputPath))
[IO.Directory]::CreateDirectory([IO.Path]::GetDirectoryName($outputFull)) | Out-Null
$lines = [IO.File]::ReadAllLines($inputFull, [Text.Encoding]::UTF8)

function Inline([string]$s) {
    $s = [Net.WebUtility]::HtmlEncode($s)
    $s = [regex]::Replace($s, '!?\[([^]]+)\]\(([^)]+)\)', {
        param($m)
        if ($m.Value.StartsWith('!')) {
            $src = $m.Groups[2].Value.Replace('\','/')
            return '<figure><img src="' + $src + '" alt="' + $m.Groups[1].Value + '"><figcaption>' + $m.Groups[1].Value + '</figcaption></figure>'
        }
        return '<a href="' + $m.Groups[2].Value + '">' + $m.Groups[1].Value + '</a>'
    })
    $s = [regex]::Replace($s, '\*\*([^*]+)\*\*', '<strong>$1</strong>')
    $s = [regex]::Replace($s, '`([^`]+)`', '<code>$1</code>')
    return $s
}

$body = [Text.StringBuilder]::new()
$inCode = $false; $inUl = $false; $inOl = $false; $inTable = $false; $para = [Collections.Generic.List[string]]::new()
function FlushPara { if ($para.Count) { [void]$body.AppendLine('<p>' + (Inline ($para -join ' ')) + '</p>'); $para.Clear() } }
function CloseLists { if ($inUl) {[void]$body.AppendLine('</ul>'); $script:inUl=$false}; if ($inOl) {[void]$body.AppendLine('</ol>'); $script:inOl=$false} }
function CloseTable { if ($inTable) {[void]$body.AppendLine('</tbody></table>'); $script:inTable=$false} }

for ($i=0; $i -lt $lines.Count; $i++) {
    $line = $lines[$i]
    if ($line -match '^```') { FlushPara; CloseLists; CloseTable; if (-not $inCode) {[void]$body.AppendLine('<pre><code>'); $inCode=$true} else {[void]$body.AppendLine('</code></pre>'); $inCode=$false}; continue }
    if ($inCode) { [void]$body.AppendLine([Net.WebUtility]::HtmlEncode($line)); continue }
    if ($line -match '^(#{1,3})\s+(.+)$') { FlushPara; CloseLists; CloseTable; $n=$matches[1].Length; [void]$body.AppendLine("<h$n>" + (Inline $matches[2]) + "</h$n>"); continue }
    if ($line -match '^!\[') { FlushPara; CloseLists; CloseTable; [void]$body.AppendLine((Inline $line)); continue }
    if ($line -match '^\|') {
        FlushPara; CloseLists
        $cells = $line.Trim('|').Split('|') | ForEach-Object { $_.Trim() }
        if ($i+1 -lt $lines.Count -and $lines[$i+1] -match '^\|(?:\s*:?-+)') {
            CloseTable; [void]$body.AppendLine('<table><thead><tr>'); foreach($c in $cells){[void]$body.AppendLine('<th>'+(Inline $c)+'</th>')}; [void]$body.AppendLine('</tr></thead><tbody>'); $inTable=$true; $i++; continue
        }
        if ($inTable) { [void]$body.AppendLine('<tr>'); foreach($c in $cells){[void]$body.AppendLine('<td>'+(Inline $c)+'</td>')}; [void]$body.AppendLine('</tr>'); continue }
    }
    CloseTable
    if ($line -match '^[-*]\s+(.+)$') { FlushPara; if ($inOl){CloseLists}; if(-not $inUl){[void]$body.AppendLine('<ul>');$inUl=$true}; [void]$body.AppendLine('<li>'+(Inline $matches[1])+'</li>'); continue }
    if ($line -match '^\d+\.\s+(.+)$') { FlushPara; if ($inUl){CloseLists}; if(-not $inOl){[void]$body.AppendLine('<ol>');$inOl=$true}; [void]$body.AppendLine('<li>'+(Inline $matches[1])+'</li>'); continue }
    if ([string]::IsNullOrWhiteSpace($line)) { FlushPara; CloseLists; continue }
    $para.Add($line.Trim())
}
FlushPara; CloseLists; CloseTable

$baseUri = ([Uri]([IO.Path]::GetDirectoryName($inputFull) + [IO.Path]::DirectorySeparatorChar)).AbsoluteUri
$html = @"
<!doctype html><html><head><meta charset="utf-8"><base href="$baseUri"><title>MultiRocket Training Pipelines</title>
<style>
@page { size: A4; margin: 18mm 17mm 18mm 17mm; @bottom-center { content: "Page " counter(page) " of " counter(pages); font: 9pt Arial; color:#64748b; } }
*{box-sizing:border-box} html,body{width:100%;overflow-wrap:anywhere} body{font:10.3pt/1.5 Arial,Segoe UI,sans-serif;color:#1f2937;margin:0} h1{font-size:25pt;color:#123b5d;margin:0 0 16pt;border-bottom:3px solid #2c7da0;padding-bottom:8pt} h2{font-size:17pt;color:#155e75;margin:22pt 0 8pt;break-after:avoid} h3{font-size:12.5pt;color:#0f4c5c;margin:15pt 0 6pt;break-after:avoid} p{margin:0 0 8pt;text-align:justify} a{color:#0369a1;text-decoration:none} code{font:9pt Consolas,monospace;background:#eef2f6;padding:1px 3px;border-radius:3px} pre{font:8.5pt/1.45 Consolas,monospace;background:#0f172a;color:#e2e8f0;padding:10pt;border-radius:5px;white-space:pre-wrap;break-inside:avoid} pre code{background:none;padding:0;color:inherit} ul,ol{margin:4pt 0 10pt 18pt;padding-left:8pt} li{margin:2pt 0} table{width:100%;border-collapse:collapse;margin:7pt 0 13pt;font-size:8.7pt;break-inside:avoid} th{background:#155e75;color:white;text-align:left;padding:5pt;border:1px solid #0e7490} td{padding:4.5pt;border:1px solid #cbd5e1;vertical-align:top} tr:nth-child(even) td{background:#f8fafc} figure{display:block;width:100%;max-width:100%;margin:9pt 0 13pt;text-align:center;break-inside:avoid;page-break-inside:avoid;overflow:hidden} figure img{display:block;max-width:100%;max-height:215mm;width:auto;height:auto;margin:auto;object-fit:contain} figcaption{font-size:8.5pt;color:#64748b;margin-top:4pt} strong{color:#0f3d56}
</style></head><body>$body</body></html>
"@
[IO.File]::WriteAllText($outputFull, $html, [Text.UTF8Encoding]::new($false))
Write-Output $outputFull
