# Docker Installation & Deployment Script

## Step 1: Install Docker Desktop (Required - One Time Only)

### Download & Install

1. **Download Docker Desktop for Windows:**
   <https://desktop.docker.com/win/main/amd64/Docker%20Desktop%20Installer.exe>

2. **Run the installer** (takes ~5 minutes)
   - Accept default settings
   - Enable WSL 2 integration (already have WSL installed ✓)
   - Restart computer if prompted

3. **Verify Docker is running:**

   ```powershell
   docker --version
   # Should show: Docker version 24.x.x or similar
   ```

---

## Step 2: Build Docker Image (Run Once After Install)

**Open PowerShell in project directory:**

```powershell
cd d:\Softwares\Antigravity\AI_Supply_Chain_Control_Tower

# Build the Docker image (~10-15 minutes first time)
docker build -t ai-supply-chain-tower .
```

**What this does:**

- Installs Python 3.10
- Installs all ML libraries (pmdarima, xgboost, prophet, etc.)
- Copies your entire app into container
- Creates a portable 2-3 GB image

---

## Step 3: Run the Container

```powershell
# Start the container
docker run -d -p 8501:8501 --name supply-chain-tower ai-supply-chain-tower

# Container will start in background
# Wait ~10 seconds for app to initialize
```

---

## Step 4: Access the App

**Open your browser to:**

```
http://localhost:8501
```

**That's it! The app is running in Docker! 🎉**

---

## Quick Commands Reference

```powershell
# Check if container is running
docker ps

# Stop the container
docker stop supply-chain-tower

# Start it again
docker start supply-chain-tower

# View logs
docker logs supply-chain-tower

# Restart container
docker restart supply-chain-tower

# Remove container (if you want to rebuild)
docker rm -f supply-chain-tower
```

---

## Alternative: Run Without Docker (If You Don't Want to Install Docker)

```powershell
# Install dependencies locally
pip install -r requirements.txt

# Run with Python
streamlit run app.py
```

Then open: <http://localhost:8501>

---

## Sharing the App With Others

### Option A: Share Docker Image File

```powershell
# Save as single 2-3 GB file
docker save ai-supply-chain-tower > supply-chain-tower.tar

# Share supply-chain-tower.tar file
# Recipient loads it:
docker load < supply-chain-tower.tar
docker run -d -p 8501:8501 ai-supply-chain-tower
```

### Option B: Push to Docker Hub (Private Registry)

```powershell
# Tag the image
docker tag ai-supply-chain-tower yourcompany/supply-chain:v1.0

# Push to registry
docker push yourcompany/supply-chain:v1.0

# Others pull and run
docker pull yourcompany/supply-chain:v1.0
docker run -d -p 8501:8501 yourcompany/supply-chain:v1.0
```

---

## Troubleshooting

**"Cannot connect to Docker daemon"**
→ Start Docker Desktop application

**"Port 8501 already in use"**
→ Change port: `docker run -d -p 8502:8501 ...`
→ Then access: <http://localhost:8502>

**Build fails with "no space left"**
→ Clean Docker: `docker system prune -a`

**App shows error after starting**
→ Check logs: `docker logs supply-chain-tower`

---

**🚀 Once Docker is installed, you'll have a single command to run everything!**
