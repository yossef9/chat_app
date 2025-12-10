const API_BASE_URL = 'http://localhost:8000'; // Update this to your backend URL

class AuthService {
    async login(email, password) {
        const formData = new FormData();
        formData.append('username', email);
        formData.append('password', password);

        try {
            const response = await fetch(`${API_BASE_URL}/users/login`, {
                method: 'POST',
                body: formData,
            });

            if (!response.ok) {
                const errorData = await response.json().catch(() => ({}));
                console.log('Backend error response:', errorData); // Debug log
                const errorMessage = errorData.detail || errorData.message || 'Login failed. Please check your credentials.';
                throw new Error(errorMessage);
            }

            const data = await response.json();

            // Get user data after successful login
            const userData = await this.getCurrentUser(data.access_token);

            return {
                access_token: data.access_token,
                user: userData
            };
        } catch (error) {
            // If it's already an Error object, throw it as is
            if (error instanceof Error) {
                throw error;
            }
            // Otherwise wrap it in an Error object
            throw new Error(error?.message || 'Login failed. Please try again.');
        }
    }

    async register(email, password) {
        const response = await fetch(`${API_BASE_URL}/users/register`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ email, password }),
        });

        if (!response.ok) {
            const errorData = await response.json().catch(() => ({}));
            const errorMessage = errorData.detail || errorData.message || 'Registration failed. Please try again.';
            throw new Error(errorMessage);
        }

        const data = await response.json();

        // Auto-login after registration
        return await this.login(email, password);
    }

    async getCurrentUser(token = null) {
        const authToken = token || localStorage.getItem('auth_token');

        if (!authToken) {
            throw new Error('No authentication token');
        }

        const response = await fetch(`${API_BASE_URL}/users/me`, {
            headers: {
                'Authorization': `Bearer ${authToken}`,
            },
        });

        if (!response.ok) {
            throw new Error('Failed to get user data');
        }

        return await response.json();
    }

    async saveApiKey(apiKey) {
        const token = localStorage.getItem('auth_token');

        if (!token) {
            throw new Error('No authentication token');
        }

        const response = await fetch(`${API_BASE_URL}/users/save-api-key`, {
            method: 'POST',
            headers: {
                'Authorization': `Bearer ${token}`,
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ openai_api_key: apiKey }),
        });

        if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.detail || 'Failed to save API key');
        }

        return await response.json();
    }

    async getAllUsers() {
        const token = localStorage.getItem('auth_token');

        if (!token) {
            throw new Error('No authentication token');
        }

        const response = await fetch(`${API_BASE_URL}/users/`, {
            headers: {
                'Authorization': `Bearer ${token}`,
            },
        });

        if (!response.ok) {
            throw new Error('Failed to get users');
        }

        return await response.json();
    }

    async removeApiKey() {
        const token = localStorage.getItem('auth_token');
        if (!token) throw new Error('No authentication token');

        const response = await fetch(`${API_BASE_URL}/users/api-key`, {
            method: 'DELETE',
            headers: {
                'Authorization': `Bearer ${token}`,
            },
        });

        if (!response.ok) {
            const errorData = await response.json().catch(() => ({}));
            throw new Error(errorData.detail || 'Failed to remove API key');
        }

        return await response.json();
    }
}

const authService = new AuthService();
export default authService;
