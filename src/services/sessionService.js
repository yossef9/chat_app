const API_BASE_URL = 'http://localhost:8000';

const getAuthHeader = () => {
    const token = localStorage.getItem('auth_token');
    if (!token) {
        console.error('No auth token found in localStorage');
        throw new Error('Not authenticated - please log in again');
    }
    return { Authorization: `Bearer ${token}` };
};

const sessionService = {
    async list() {
        const response = await fetch(`${API_BASE_URL}/chat/sessions`, {
            method: 'GET',
            headers: {
                ...getAuthHeader(),
            },
        });
        
        if (!response.ok) {
            if (response.status === 401) {
                // Token expired or invalid - clear token and redirect to login
                localStorage.removeItem('auth_token');
                window.location.href = '/login';
                throw new Error('Session expired - please log in again');
            }
            const err = await response.json().catch(() => ({}));
            throw new Error(err.detail || 'Failed to list sessions');
        }
        return await response.json();
    },

    async create(name, documentIds = []) {
        // Backend route accepts query/body depending on implementation; send JSON body
        const response = await fetch(`${API_BASE_URL}/chat/sessions`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                ...getAuthHeader(),
            },
            body: JSON.stringify({ name, document_ids: documentIds }),
        });
        if (!response.ok) {
            if (response.status === 401) {
                localStorage.removeItem('auth_token');
                window.location.href = '/login';
                throw new Error('Session expired - please log in again');
            }
            const err = await response.json().catch(() => ({}));
            throw new Error(err.detail || 'Failed to create session');
        }
        return await response.json();
    },

    async get(sessionId) {
        const response = await fetch(`${API_BASE_URL}/chat/sessions/${sessionId}`, {
            method: 'GET',
            headers: {
                ...getAuthHeader(),
            },
        });
        if (!response.ok) {
            if (response.status === 401) {
                localStorage.removeItem('auth_token');
                window.location.href = '/login';
                throw new Error('Session expired - please log in again');
            }
            const err = await response.json().catch(() => ({}));
            throw new Error(err.detail || 'Failed to fetch session');
        }
        const session = await response.json();
        
        // Also fetch messages for this session
        const messages = await this.getMessages(sessionId);
        session.messages = messages;
        
        return session;
    },

    async getMessages(sessionId) {
        const response = await fetch(`${API_BASE_URL}/chat/sessions/${sessionId}/messages`, {
            method: 'GET',
            headers: {
                ...getAuthHeader(),
            },
        });
        if (!response.ok) {
            if (response.status === 401) {
                localStorage.removeItem('auth_token');
                window.location.href = '/login';
                throw new Error('Session expired - please log in again');
            }
            const err = await response.json().catch(() => ({}));
            throw new Error(err.detail || 'Failed to fetch messages');
        }
        return await response.json();
    },

    async appendMessage(sessionId, message) {
        const response = await fetch(`${API_BASE_URL}/chat/sessions/${sessionId}/message`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                ...getAuthHeader(),
            },
            body: JSON.stringify(message),
        });
        if (!response.ok) {
            if (response.status === 401) {
                localStorage.removeItem('auth_token');
                window.location.href = '/login';
                throw new Error('Session expired - please log in again');
            }
            const err = await response.json().catch(() => ({}));
            throw new Error(err.detail || 'Failed to append message');
        }
        return await response.json();
    },

    async remove(sessionId) {
        const response = await fetch(`${API_BASE_URL}/chat/sessions/${sessionId}`, {
            method: 'DELETE',
            headers: {
                ...getAuthHeader(),
            },
        });
        if (!response.ok) {
            if (response.status === 401) {
                localStorage.removeItem('auth_token');
                window.location.href = '/login';
                throw new Error('Session expired - please log in again');
            }
            const err = await response.json().catch(() => ({}));
            throw new Error(err.detail || 'Failed to delete session');
        }
        return { status: 'ok' };
    },

    async clearMessages(sessionId) {
        const response = await fetch(`${API_BASE_URL}/chat/sessions/${sessionId}/messages`, {
            method: 'DELETE',
            headers: {
                ...getAuthHeader(),
            },
        });
        if (!response.ok) {
            if (response.status === 401) {
                localStorage.removeItem('auth_token');
                window.location.href = '/login';
                throw new Error('Session expired - please log in again');
            }
            const err = await response.json().catch(() => ({}));
            throw new Error(err.detail || 'Failed to clear messages');
        }
        return { status: 'ok' };
    },

    async export(sessionId, format = 'json') {
        const session = await this.get(sessionId);

        if (format === 'markdown') {
            // Export as Markdown
            let markdown = `# ${session.name || 'Chat Session'}\n\n`;
            markdown += `**Session ID:** ${session.id}\n`;
            markdown += `**Created:** ${new Date(session.created_at).toLocaleString()}\n`;
            markdown += `**Updated:** ${new Date(session.updated_at).toLocaleString()}\n`;
            markdown += `**Documents:** ${(session.document_ids || []).length}\n\n`;
            markdown += `---\n\n`;

            session.messages?.forEach((msg, index) => {
                const role = msg.role === 'user' ? '**User:**' : (msg.role === 'assistant' ? '**Assistant:**' : '**System:**');
                markdown += `${role}\n${msg.text}\n\n`;
                if (msg.sources && msg.sources.length > 0) {
                    markdown += `*Sources:*\n`;
                    msg.sources.forEach(source => {
                        markdown += `- ${source.filename} (chunk #${source.chunk_index})\n`;
                    });
                    markdown += `\n`;
                }
            });

            const blob = new Blob([markdown], { type: 'text/markdown' });
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = `${session.name || 'session'}-${session.id}.md`;
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);
            URL.revokeObjectURL(url);
        } else {
            // Export as JSON
            const blob = new Blob([JSON.stringify(session, null, 2)], { type: 'application/json' });
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = `${session.name || 'session'}-${session.id}.json`;
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);
            URL.revokeObjectURL(url);
        }
    },
};

export default sessionService;


