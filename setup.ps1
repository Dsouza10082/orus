# Orus API Setup Script for Windows
# Automated setup for Orus API with Docker Compose
# Compatible with PowerShell 5.1+ and PowerShell Core

# Error handling
$ErrorActionPreference = "Stop"

# Colors for output
function Write-Info {
    param([string]$Message)
    Write-Host "[INFO] " -ForegroundColor Blue -NoNewline
    Write-Host $Message
}

function Write-Success {
    param([string]$Message)
    Write-Host "[SUCCESS] " -ForegroundColor Green -NoNewline
    Write-Host $Message
}

function Write-Warning {
    param([string]$Message)
    Write-Host "[WARNING] " -ForegroundColor Yellow -NoNewline
    Write-Host $Message
}

function Write-Error-Custom {
    param([string]$Message)
    Write-Host "[ERROR] " -ForegroundColor Red -NoNewline
    Write-Host $Message
}

function Write-Header {
    param([string]$Message)
    Write-Host ""
    Write-Host "========================================" -ForegroundColor Blue
    Write-Host $Message -ForegroundColor Blue
    Write-Host "========================================" -ForegroundColor Blue
    Write-Host ""
}

# Check if command exists
function Test-CommandExists {
    param([string]$Command)
    $null -ne (Get-Command $Command -ErrorAction SilentlyContinue)
}

# Wait for service to be healthy
function Wait-ForService {
    param(
        [string]$Url,
        [int]$MaxAttempts
    )
    
    Write-Info "Waiting for service at $Url..."
    
    for ($attempt = 1; $attempt -le $MaxAttempts; $attempt++) {
        try {
            $response = Invoke-WebRequest -Uri $Url -Method Get -TimeoutSec 2 -UseBasicParsing -ErrorAction SilentlyContinue
            if ($response.StatusCode -eq 200) {
                Write-Success "Service is ready!"
                return $true
            }
        }
        catch {
            # Service not ready yet
        }
        
        Write-Host "." -NoNewline
        Start-Sleep -Seconds 2
    }
    
    Write-Host ""
    Write-Error-Custom "Service did not become ready after $MaxAttempts attempts"
    return $false
}

# Main setup function
function Main {
    Write-Header "Orus API Setup Script"
    
    # Step 1: Check prerequisites
    Write-Info "Checking prerequisites..."
    
    if (-not (Test-CommandExists "docker")) {
        Write-Error-Custom "Docker is not installed. Please install Docker Desktop first."
        Write-Info "Visit: https://docs.docker.com/desktop/install/windows-install/"
        exit 1
    }
    Write-Success "Docker is installed"
    
    # Check if Docker is running
    try {
        docker ps | Out-Null
        Write-Success "Docker is running"
    }
    catch {
        Write-Error-Custom "Docker is not running. Please start Docker Desktop first."
        exit 1
    }
    
    if (-not (Test-CommandExists "docker-compose")) {
        # Check for docker compose (without hyphen)
        try {
            docker compose version | Out-Null
            Write-Success "Docker Compose is installed"
        }
        catch {
            Write-Error-Custom "Docker Compose is not installed. Please install Docker Compose first."
            Write-Info "Visit: https://docs.docker.com/compose/install/"
            exit 1
        }
    }
    else {
        Write-Success "Docker Compose is installed"
    }
    
    if (-not (Test-CommandExists "curl")) {
        Write-Warning "curl is not available. Using Invoke-WebRequest instead."
    }
    else {
        Write-Success "curl is installed"
    }
    
    # Step 2: Check if docker-compose.yml exists
    if (-not (Test-Path "docker-compose.yml")) {
        Write-Error-Custom "docker-compose.yml not found in current directory"
        Write-Info "Please run this script from the project root directory"
        Write-Info "Current directory: $(Get-Location)"
        exit 1
    }
    Write-Success "docker-compose.yml found"
    
    # Step 3: Stop existing containers
    Write-Header "Step 1: Stopping existing containers (if any)"
    try {
        if (Test-CommandExists "docker-compose") {
            docker-compose down 2>$null
        }
        else {
            docker compose down 2>$null
        }
        Write-Success "Cleaned up existing containers"
    }
    catch {
        Write-Info "No existing containers to stop"
    }
    
    # Step 4: Start services
    Write-Header "Step 2: Starting Orus API services"
    Write-Info "This may take a few minutes on first run..."
    
    try {
        if (Test-CommandExists "docker-compose") {
            docker-compose up -d
        }
        else {
            docker compose up -d
        }
        Write-Success "Services started"
    }
    catch {
        Write-Error-Custom "Failed to start services"
        Write-Info "Error: $_"
        exit 1
    }
    
    # Step 5: Wait for Ollama service
    Write-Header "Step 3: Waiting for Ollama service"
    if (-not (Wait-ForService -Url "http://localhost:11434/api/tags" -MaxAttempts 30)) {
        Write-Error-Custom "Ollama service failed to start"
        Write-Info "Check logs with: docker-compose logs ollama"
        exit 1
    }
    
    # Step 6: Wait for Orus API
    Write-Header "Step 4: Waiting for Orus API"
    if (-not (Wait-ForService -Url "http://localhost:8081/orus-api/v1/system-info" -MaxAttempts 30)) {
        Write-Error-Custom "Orus API failed to start"
        Write-Info "Check logs with: docker-compose logs orus-api"
        exit 1
    }
    
    # Step 7: Verify services
    Write-Header "Step 5: Verifying services"
    
    Write-Info "Testing Orus API..."
    try {
        $response = Invoke-RestMethod -Uri "http://localhost:8081/orus-api/v1/system-info" -Method Get -UseBasicParsing
        if ($response.success -eq $true) {
            Write-Success "Orus API is responding correctly"
        }
        else {
            Write-Warning "Orus API response is unexpected"
        }
    }
    catch {
        Write-Warning "Could not verify Orus API response"
    }
    
    Write-Info "Testing Ollama service..."
    try {
        $response = Invoke-WebRequest -Uri "http://localhost:11434/api/tags" -Method Get -UseBasicParsing
        if ($response.StatusCode -eq 200) {
            Write-Success "Ollama service is responding correctly"
        }
        else {
            Write-Warning "Ollama service response is unexpected"
        }
    }
    catch {
        Write-Warning "Could not verify Ollama service response"
    }
    
    # Step 8: Display next steps
    Write-Header "Setup Complete!"
    
    Write-Host ""
    Write-Host "✓ Orus API is running on: " -ForegroundColor Green -NoNewline
    Write-Host "http://localhost:8081" -ForegroundColor Blue
    Write-Host "✓ Ollama service is running on: " -ForegroundColor Green -NoNewline
    Write-Host "http://localhost:11434" -ForegroundColor Blue
    Write-Host ""
    
    Write-Warning "IMPORTANT: You must download models before using the API!"
    Write-Host ""
    Write-Host "Next Steps:" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "1. Download the embedding model (REQUIRED for /embed-text):"
    Write-Host "   Using curl:" -ForegroundColor Gray
    Write-Host '   curl -X POST http://localhost:8081/orus-api/v1/ollama-pull-model \' -ForegroundColor Yellow
    Write-Host '     -H "Content-Type: application/json" \' -ForegroundColor Yellow
    Write-Host '     -d "{\"name\": \"nomic-embed-text\"}"' -ForegroundColor Yellow
    Write-Host ""
    Write-Host "   Using PowerShell:" -ForegroundColor Gray
    Write-Host '   Invoke-RestMethod -Uri "http://localhost:8081/orus-api/v1/ollama-pull-model" `' -ForegroundColor Yellow
    Write-Host '     -Method Post -ContentType "application/json" `' -ForegroundColor Yellow
    Write-Host '     -Body ''{"name": "nomic-embed-text"}''' -ForegroundColor Yellow
    Write-Host ""
    Write-Host "2. Download an LLM model (REQUIRED for /call-llm):"
    Write-Host "   Choose one of these popular models:"
    Write-Host ""
    Write-Host "   • Llama 3.1 8B (~4.7GB, recommended):"
    Write-Host '     curl -X POST http://localhost:8081/orus-api/v1/ollama-pull-model \' -ForegroundColor Yellow
    Write-Host '       -H "Content-Type: application/json" \' -ForegroundColor Yellow
    Write-Host '       -d "{\"name\": \"llama3.1:8b\"}"' -ForegroundColor Yellow
    Write-Host ""
    Write-Host "   • Mistral 7B (~4.1GB, fast):"
    Write-Host '     curl -X POST http://localhost:8081/orus-api/v1/ollama-pull-model \' -ForegroundColor Yellow
    Write-Host '       -H "Content-Type: application/json" \' -ForegroundColor Yellow
    Write-Host '       -d "{\"name\": \"mistral:7b\"}"' -ForegroundColor Yellow
    Write-Host ""
    Write-Host "   • Phi-3 Mini (~2.2GB, lightweight):"
    Write-Host '     curl -X POST http://localhost:8081/orus-api/v1/ollama-pull-model \' -ForegroundColor Yellow
    Write-Host '       -H "Content-Type: application/json" \' -ForegroundColor Yellow
    Write-Host '       -d "{\"name\": \"phi3:mini\"}"' -ForegroundColor Yellow
    Write-Host ""
    Write-Host "3. Verify downloaded models:"
    Write-Host "   curl http://localhost:8081/orus-api/v1/ollama-model-list" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "   Or with PowerShell:" -ForegroundColor Gray
    Write-Host '   Invoke-RestMethod -Uri "http://localhost:8081/orus-api/v1/ollama-model-list"' -ForegroundColor Yellow
    Write-Host ""
    Write-Host "4. Test the API:"
    Write-Host '   $body = @{' -ForegroundColor Yellow
    Write-Host '     body = @{' -ForegroundColor Yellow
    Write-Host '       model = "llama3.1:8b"' -ForegroundColor Yellow
    Write-Host '       stream = $false' -ForegroundColor Yellow
    Write-Host '       messages = @(' -ForegroundColor Yellow
    Write-Host '         @{role = "user"; content = "Hello!"}' -ForegroundColor Yellow
    Write-Host '       )' -ForegroundColor Yellow
    Write-Host '     }' -ForegroundColor Yellow
    Write-Host '   } | ConvertTo-Json -Depth 10' -ForegroundColor Yellow
    Write-Host '   Invoke-RestMethod -Uri "http://localhost:8081/orus-api/v1/call-llm" `' -ForegroundColor Yellow
    Write-Host '     -Method Post -ContentType "application/json" -Body $body' -ForegroundColor Yellow
    Write-Host ""
    Write-Host "Useful Commands:" -ForegroundColor Cyan
    Write-Host "  • View logs:        " -NoNewline
    Write-Host "docker-compose logs -f" -ForegroundColor Blue
    Write-Host "  • Stop services:    " -NoNewline
    Write-Host "docker-compose down" -ForegroundColor Blue
    Write-Host "  • Restart services: " -NoNewline
    Write-Host "docker-compose restart" -ForegroundColor Blue
    Write-Host ""
    Write-Host "Documentation:" -ForegroundColor Cyan
    Write-Host "  • API Reference: " -NoNewline
    Write-Host ".\API.md" -ForegroundColor Blue
    Write-Host "  • README:        " -NoNewline
    Write-Host ".\README.md" -ForegroundColor Blue
    Write-Host ""
    
    Write-Success "Setup script completed successfully!"
}

# Run main function
try {
    Main
}
catch {
    Write-Error-Custom "An error occurred during setup"
    Write-Host "Error details: $_" -ForegroundColor Red
    exit 1
}
