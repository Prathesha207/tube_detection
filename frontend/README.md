# DHTX-V2

AI Vision Monitoring System built with Next.js, TypeScript, Zustand, and WebSocket streaming.

## Features

- Live camera streaming
- Video upload processing
- AI inference detection
- ROI tools
- Recording support
- Real-time overlays
- WebSocket communication
- Processed and raw frame visualization

## Tech Stack

- Next.js
- React
- TypeScript
- Tailwind CSS
- Zustand
- WebSocket
- Electron

## Installation

```bash
npm install
```

## Environment Variables

Create `.env` file:

```env
NEXT_PUBLIC_BASE_URL=http://127.0.0.1:8000
NEXT_PUBLIC_WS_BASE_URL=ws://127.0.0.1:8000
```

## Run Development Server

```bash
npm run dev
```

## Build Project

```bash
npm run build
```

## Start Production

```bash
npm start
```

## Git Setup

```bash
git init
git add .
git commit -m "Initial commit"
```

## Project Structure

```bash
src/
 ├── app/
 ├── components/
 ├── hooks/
 ├── services/
 ├── store/
 ├── utils/
 └── styles/
```

## Notes

- Restart the frontend after updating `.env`
- Backend server should run on port `8000`
- Make sure WebSocket and API server are running before starting inference
