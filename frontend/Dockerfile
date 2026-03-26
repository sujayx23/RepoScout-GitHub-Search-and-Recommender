FROM node:20-alpine AS build

WORKDIR /app

ARG VITE_API_URL=http://localhost:8000
ENV VITE_API_URL=$VITE_API_URL

COPY package*.json ./
RUN npm ci

COPY . .
RUN npm run build

# Serve with nginx
FROM nginx:stable-alpine

COPY --from=build /app/dist /usr/share/nginx/html

# nginx config: serve SPA, proxy /api to backend
COPY nginx.conf /etc/nginx/conf.d/default.conf

EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]
