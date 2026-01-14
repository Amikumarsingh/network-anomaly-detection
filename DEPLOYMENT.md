# Deployment Guide

## Option 1: Render (Recommended - Free Tier)

### Backend Deployment:
1. Go to [render.com](https://render.com) and sign up
2. Click "New +" → "Web Service"
3. Connect your GitHub repository
4. Configure:
   - **Name**: anomaly-detection-backend
   - **Environment**: Python 3
   - **Build Command**: `pip install -r backend/requirements.txt`
   - **Start Command**: `python backend/main.py`
   - **Plan**: Free
5. Click "Create Web Service"
6. Copy your backend URL (e.g., `https://anomaly-detection-backend.onrender.com`)

### Frontend Deployment:
1. In Render, click "New +" → "Static Site"
2. Connect same GitHub repository
3. Configure:
   - **Name**: anomaly-detection-frontend
   - **Publish Directory**: `frontend`
   - **Plan**: Free
4. Before deploying, update `frontend/index.html`:
   - Replace `YOUR_BACKEND_URL` with your actual backend URL
5. Click "Create Static Site"

## Option 2: Railway (Simple)

1. Go to [railway.app](https://railway.app)
2. Click "New Project" → "Deploy from GitHub repo"
3. Select your repository
4. Railway will auto-detect and deploy
5. Add environment variables if needed

## Option 3: Vercel (Frontend) + Render (Backend)

### Backend on Render (same as Option 1)

### Frontend on Vercel:
1. Go to [vercel.com](https://vercel.com)
2. Click "Add New" → "Project"
3. Import your GitHub repository
4. Configure:
   - **Framework Preset**: Other
   - **Root Directory**: `frontend`
5. Update `frontend/index.html` with backend URL
6. Click "Deploy"

## Post-Deployment Steps:

1. **Update Frontend URLs**:
   Edit `frontend/index.html` and replace:
   ```javascript
   const API_URL = 'https://YOUR-BACKEND-URL.onrender.com';
   const WS_URL = 'wss://YOUR-BACKEND-URL.onrender.com/ws';
   ```

2. **Test Your Deployment**:
   - Visit your frontend URL
   - Click "Train Models" (wait 60 seconds)
   - Click "Start Detection"
   - Watch live anomaly detection!

3. **Commit and Push**:
   ```bash
   git add .
   git commit -m "Add deployment configuration"
   git push origin main
   ```

## Quick Deploy Commands:

```bash
# Commit deployment files
git add Dockerfile render.yaml vercel.json frontend/index.html
git commit -m "Add deployment configuration"
git push origin main
```

## Environment Variables (if needed):
- `PORT`: 10000 (Render default)
- `HOST`: 0.0.0.0

## Notes:
- Free tier may have cold starts (30-60 seconds)
- WebSocket connections work on all platforms
- HTTPS/WSS automatically configured
