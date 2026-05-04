FROM node:20-alpine

WORKDIR /app

# Copy package files first (layer cache — only re-runs npm install if package.json changed)
COPY package.json ./

# Install dependencies
RUN npm install --omit=dev --no-audit --no-fund

# Copy source
COPY server.js ./

# Expose port (platforms override this via PORT env var)
EXPOSE 3000

# Health check so the platform knows the container is ready
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD wget -qO- http://localhost:${PORT:-3000}/health || exit 1

CMD ["node", "server.js"]
