const API_BASE_URL = 'http://localhost:8000';

const getAuthHeader = () => {
    const token = localStorage.getItem('auth_token');
    if (!token) throw new Error('Not authenticated');
    return { Authorization: `Bearer ${token}` };
};

const documentService = {
    async upload(file, onProgress) {
        const formData = new FormData();
        formData.append('file', file);

        const response = await fetch(`${API_BASE_URL}/documents`, {
            method: 'POST',
            headers: {
                ...getAuthHeader(),
            },
            body: formData,
        });

        if (!response.ok) {
            const err = await response.json().catch(() => ({}));
            throw new Error(err.detail || 'Upload failed');
        }

        return await response.json();
    },

    async listMine() {
        const response = await fetch(`${API_BASE_URL}/documents/me`, {
            headers: {
                ...getAuthHeader(),
            },
        });
        if (!response.ok) throw new Error('Failed to load documents');
        return await response.json();
    },

    async remove(id) {
        const response = await fetch(`${API_BASE_URL}/documents/${id}`, {
            method: 'DELETE',
            headers: {
                ...getAuthHeader(),
            },
        });
        if (!response.ok) {
            const err = await response.json().catch(() => ({}));
            throw new Error(err.detail || 'Delete failed');
        }
        return await response.json();
    },
};

export default documentService;


