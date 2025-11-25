#!/bin/bash
# build_and_test.sh - Build Docker image and run tests

echo "🔨 Building AI Supply Chain Control Tower Docker Image..."

# Build the image
docker build -t ai-supply-chain-tower:latest .

if [ $? -eq 0 ]; then
    echo "✅ Docker image built successfully!"
    
    echo "📏 Image size:"
    docker images ai-supply-chain-tower:latest
    
    echo ""
    echo "🧪 Running quick test..."
    
    # Run container in detached mode
    docker run -d --name supply-chain-test -p 8501:8501 ai-supply-chain-tower:latest
    
    # Wait for app to start
    echo "Waiting for app to start (30 seconds)..."
    sleep 30
    
    # Check if app is responding
    if curl -f http://localhost:8501/_stcore/health > /dev/null 2>&1; then
        echo "✅ App is running!"
        echo "🌐 Access at: http://localhost:8501"
        echo ""
        echo "To stop the test container:"
        echo "  docker stop supply-chain-test"
        echo "  docker rm supply-chain-test"
    else
        echo "❌ App health check failed"
        echo "Check logs with: docker logs supply-chain-test"
        docker stop supply-chain-test
        docker rm supply-chain-test
        exit 1
    fi
    
else
    echo "❌ Docker build failed!"
    exit 1
fi
