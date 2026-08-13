import axios from 'axios';

const envApiUrl = (import.meta as unknown as { env?: { VITE_API_URL?: string } }).env?.VITE_API_URL;
const API_URL = envApiUrl || 'http://localhost:8000/api/v1';

export const apiClient = axios.create({
  baseURL: API_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

apiClient.interceptors.request.use((config) => {
  const token = localStorage.getItem('auth_token');
  if (token) {
    config.headers.Authorization = `Bearer ${token}`;
  }
  return config;
});

apiClient.interceptors.response.use(
  (response) => response,
  (error) => {
    if (error.response?.status === 401) {
      console.warn('Unauthorized access, token may be expired.');
    }
    return Promise.reject(error);
  }
);

