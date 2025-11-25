# 🚀 Quick Start - Get Docker & Run

## ⚠️ Docker Not Installed Yet

Your system has WSL2 (✓) but needs Docker Desktop.

---

## 📥 Step 1: Install Docker Desktop (5 Minutes)

### Direct Download Link

**Click here:** <https://desktop.docker.com/win/main/amd64/Docker%20Desktop%20Installer.exe>

### Installation Steps

1. Run the downloaded installer
2. Accept defaults (WSL 2 backend - you already have WSL ✓)
3. Restart computer if prompted
4. Start Docker Desktop application

### Verify Docker is Working

```powershell
docker --version
# Should show: Docker version 24.x.x
```

---

## 🚀 Step 2: One-Click Deploy (After Docker Installed)

### Method A: Double-Click `DEPLOY.bat`

Just double-click the **DEPLOY.bat** file in this folder!

It will:

- ✅ Check Docker is installed
- ✅ Build the Docker image (~10-15 min first time)
- ✅ Start the container
- ✅ Open browser to <http://localhost:8501>

### Method B: Manual Commands

```powershell
cd d:\Softwares\Antigravity\AI_Supply_Chain_Control_Tower

# Build image
docker build -t ai-supply-chain-tower .

# Run container
docker run -d -p 8501:8501 --name supply-chain-tower ai-supply-chain-tower

# Open browser to:
# http://localhost:8501
```

---

## 🌐 Your Localhost Link

Once running, access at:

```
http://localhost:8501
```

**Container will run in background until you stop it.**

---

## 🎮 Container Management

```powershell
# Check status
docker ps

# Stop (keeps data)
docker stop supply-chain-tower

# Start again
docker start supply-chain-tower

# Restart
docker restart supply-chain-tower

# View logs
docker logs supply-chain-tower

# Remove completely
docker rm -f supply-chain-tower
```

---

## ⚡ Alternative: Skip Docker (Faster for Testing)

If you don't want to install Docker right now:

```powershell
# Install Python dependencies
pip install -r requirements.txt

# Run with Streamlit
streamlit run app.py
```

Then open: <http://localhost:8501>

---

## 📦 What's Inside the Docker Container

- ✅ Python 3.10
- ✅ All ML libraries (pmdarima, xgboost, prophet, shap, etc.)
- ✅ Your entire supply chain app
- ✅ Demo data (91,250 rows)
- ✅ Intelligent modules (forecasting, inventory, logistics, Monte Carlo)
- ✅ 100% offline operation (no external APIs)

**Image size:** ~2.1 GB  
**Build time:** 10-15 minutes (first time only)  
**Run time:** Instant (after first build)

---

## 🎯 Next Steps

1. **Install Docker Desktop** (link above)
2. **Double-click DEPLOY.bat** or run manual commands
3. **Wait ~15 minutes** for first build
4. **Open <http://localhost:8501>**
5. **Load demo data** and explore intelligent features!

---

## 🆘 Troubleshooting

**"Docker is not recognized"**
→ Install Docker Desktop (see Step 1)

**"Cannot connect to Docker daemon"**
→ Start Docker Desktop application

**"Port 8501 already in use"**
→ Change port: `docker run -d -p 8502:8501 ...`
→ Then use: <http://localhost:8502>

**Build takes too long**
→ This is normal for first build (installing all ML libraries)
→ Subsequent builds are instant

---

**🚀 See DOCKER_SETUP.md for detailed documentation.**
