#!/bin/bash

# Orus API Setup Script
# Automated setup for Orus API with Docker Compose
# Compatible with Linux and macOS

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Print colored messages
print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_header() {
    echo ""
    echo -e "${BLUE}========================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}========================================${NC}"
    echo ""
}

# Check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Wait for a service to be healthy
wait_for_service() {
    local url=$1
    local max_attempts=$2
    local attempt=1
    
    print_info "Waiting for service at $url..."
    
    while [ $attempt -le $max_attempts ]; do
        if curl -s -f "$url" > /dev/null 2>&1; then
            print_success "Service is ready!"
            return 0
        fi
        
        echo -n "."
        sleep 2
        attempt=$((attempt + 1))
    done
    
    echo ""
    print_error "Service did not become ready after $max_attempts attempts"
    return 1
}

# Main setup function
main() {
    print_header "Orus API Setup Script"
    
    # Step 1: Check prerequisites
    print_info "Checking prerequisites..."
    
    if ! command_exists docker; then
        print_error "Docker is not installed. Please install Docker first."
        print_info "Visit: https://docs.docker.com/get-docker/"
        exit 1
    fi
    print_success "Docker is installed"
    
    if ! command_exists docker-compose && ! docker compose version > /dev/null 2>&1; then
        print_error "Docker Compose is not installed. Please install Docker Compose first."
        print_info "Visit: https://docs.docker.com/compose/install/"
        exit 1
    fi
    print_success "Docker Compose is installed"
    
    if ! command_exists curl; then
        print_error "curl is not installed. Please install curl first."
        exit 1
    fi
    print_success "curl is installed"
    
    # Step 2: Check if docker-compose.yml exists
    if [ ! -f "docker-compose.yml" ]; then
        print_error "docker-compose.yml not found in current directory"
        print_info "Please run this script from the project root directory"
        exit 1
    fi
    print_success "docker-compose.yml found"
    
    # Step 3: Stop existing containers
    print_header "Step 1: Stopping existing containers (if any)"
    docker-compose down 2>/dev/null || docker compose down 2>/dev/null || true
    print_success "Cleaned up existing containers"
    
    # Step 4: Start services
    print_header "Step 2: Starting Orus API services"
    print_info "This may take a few minutes on first run..."
    
    if command_exists docker-compose; then
        docker-compose up -d
    else
        docker compose up -d
    fi
    
    print_success "Services started"
    
    # Step 5: Wait for Ollama service
    print_header "Step 3: Waiting for Ollama service"
    if ! wait_for_service "http://localhost:11434/api/tags" 30; then
        print_error "Ollama service failed to start"
        print_info "Check logs with: docker-compose logs ollama"
        exit 1
    fi
    
    # Step 6: Wait for Orus API
    print_header "Step 4: Waiting for Orus API"
    if ! wait_for_service "http://localhost:8081/orus-api/v1/system-info" 30; then
        print_error "Orus API failed to start"
        print_info "Check logs with: docker-compose logs orus-api"
        exit 1
    fi
    
    # Step 7: Verify services
    print_header "Step 5: Verifying services"
    
    print_info "Testing Orus API..."
    if curl -s http://localhost:8081/orus-api/v1/system-info | grep -q "success"; then
        print_success "Orus API is responding correctly"
    else
        print_warning "Orus API response is unexpected"
    fi
    
    print_info "Testing Ollama service..."
    if curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
        print_success "Ollama service is responding correctly"
    else
        print_warning "Ollama service response is unexpected"
    fi
    
    # Step 8: Display next steps
    print_header "Setup Complete!"
    
    echo ""
    echo -e "${GREEN}✓${NC} Orus API is running on: ${BLUE}http://localhost:8081${NC}"
    echo -e "${GREEN}✓${NC} Ollama service is running on: ${BLUE}http://localhost:11434${NC}"
    echo ""
    
    print_warning "IMPORTANT: You must download models before using the API!"
    echo ""
    echo "Next Steps:"
    echo ""
    echo "1. Download the embedding model (REQUIRED for /embed-text):"
    echo -e "   ${YELLOW}curl -X POST http://localhost:8081/orus-api/v1/ollama-pull-model \\${NC}"
    echo -e "   ${YELLOW}     -H 'Content-Type: application/json' \\${NC}"
    echo -e "   ${YELLOW}     -d '{\"name\": \"nomic-embed-text\"}'${NC}"
    echo ""
    echo "2. Download an LLM model (REQUIRED for /call-llm):"
    echo "   Choose one of these popular models:"
    echo ""
    echo "   • Llama 3.1 8B (~4.7GB, recommended):"
    echo -e "     ${YELLOW}curl -X POST http://localhost:8081/orus-api/v1/ollama-pull-model \\${NC}"
    echo -e "     ${YELLOW}       -H 'Content-Type: application/json' \\${NC}"
    echo -e "     ${YELLOW}       -d '{\"name\": \"llama3.1:8b\"}'${NC}"
    echo ""
    echo "   • Mistral 7B (~4.1GB, fast):"
    echo -e "     ${YELLOW}curl -X POST http://localhost:8081/orus-api/v1/ollama-pull-model \\${NC}"
    echo -e "     ${YELLOW}       -H 'Content-Type: application/json' \\${NC}"
    echo -e "     ${YELLOW}       -d '{\"name\": \"mistral:7b\"}'${NC}"
    echo ""
    echo "   • Phi-3 Mini (~2.2GB, lightweight):"
    echo -e "     ${YELLOW}curl -X POST http://localhost:8081/orus-api/v1/ollama-pull-model \\${NC}"
    echo -e "     ${YELLOW}       -H 'Content-Type: application/json' \\${NC}"
    echo -e "     ${YELLOW}       -d '{\"name\": \"phi3:mini\"}'${NC}"
    echo ""
    echo "3. Verify downloaded models:"
    echo -e "   ${YELLOW}curl http://localhost:8081/orus-api/v1/ollama-model-list${NC}"
    echo ""
    echo "4. Test the API:"
    echo -e "   ${YELLOW}curl -X POST http://localhost:8081/orus-api/v1/call-llm \\${NC}"
    echo -e "   ${YELLOW}     -H 'Content-Type: application/json' \\${NC}"
    echo -e "   ${YELLOW}     -d '{\"body\":{\"model\":\"llama3.1:8b\",\"stream\":false,\"messages\":[{\"role\":\"user\",\"content\":\"Hello!\"}]}}'${NC}"
    echo ""
    echo "Useful Commands:"
    echo -e "  • View logs:        ${BLUE}docker-compose logs -f${NC}"
    echo -e "  • Stop services:    ${BLUE}docker-compose down${NC}"
    echo -e "  • Restart services: ${BLUE}docker-compose restart${NC}"
    echo ""
    echo "Documentation:"
    echo -e "  • API Reference: ${BLUE}./API.md${NC}"
    echo -e "  • README:        ${BLUE}./README.md${NC}"
    echo ""
    
    print_success "Setup script completed successfully!"
}

# Run main function
main "$@"
