const API_BASE_URL = 'http://localhost:8000';

const getAuthHeader = () => {
    const token = localStorage.getItem('auth_token');
    if (!token) throw new Error('Not authenticated');
    return { Authorization: `Bearer ${token}` };
};

const chatService = {
    async ask(question, documentIds) {
        const response = await fetch(`${API_BASE_URL}/chat/ask`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                ...getAuthHeader(),
            },
            body: JSON.stringify({ question, document_ids: documentIds })
        });

        if (!response.ok) {
            const err = await response.json().catch(() => ({}));
            throw new Error(err.detail || 'Chat failed');
        }

        return await response.json();
    }
};

export default chatService;


