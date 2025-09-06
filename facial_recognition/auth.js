// Mock user data for demonstration
const mockUsers = [
    { email: 'admin@facerecog.com', password: 'admin123', name: 'Admin User', role: 'admin' },
    { email: 'user@facerecog.com', password: 'user123', name: 'John Doe', role: 'user' }
];

function handleLogin(event) {
    event.preventDefault();
    
    const formData = new FormData(event.target);
    const email = formData.get('email');
    const password = formData.get('password');
    
    // Mock authentication
    const user = mockUsers.find(u => u.email === email && u.password === password);
    
    if (user) {
        // Store user session
        localStorage.setItem('currentUser', JSON.stringify(user));
        
        // Show success message
        showMessage('Login successful! Redirecting...', 'success');
        
        // Redirect to dashboard
        setTimeout(() => {
            window.location.href = 'dashboard.html';
        }, 1500);
    } else {
        showMessage('Invalid email or password', 'error');
    }
}

function handleSignup(event) {
    event.preventDefault();
    
    const formData = new FormData(event.target);
    const name = formData.get('name');
    const email = formData.get('email');
    const password = formData.get('password');
    const role = formData.get('role');
    
    // Check if user already exists
    const existingUser = mockUsers.find(u => u.email === email);
    
    if (existingUser) {
        showMessage('User with this email already exists', 'error');
        return;
    }
    
    // Create new user
    const newUser = { email, password, name, role };
    mockUsers.push(newUser);
    
    // Store user session
    localStorage.setItem('currentUser', JSON.stringify(newUser));
    
    showMessage('Account created successfully! Redirecting...', 'success');
    
    setTimeout(() => {
        window.location.href = 'dashboard.html';
    }, 1500);
}

function handleForgotPassword(event) {
    event.preventDefault();
    
    const formData = new FormData(event.target);
    const email = formData.get('email');
    
    // Mock password reset
    const user = mockUsers.find(u => u.email === email);
    
    if (user) {
        showMessage('Password reset link sent to your email!', 'success');
    } else {
        showMessage('No account found with this email', 'error');
    }
}

function socialLogin(provider) {
    showMessage(`${provider} login would be implemented here`, 'info');
    
    // Mock social login success
    setTimeout(() => {
        const mockSocialUser = {
            email: `user@${provider}.com`,
            name: `${provider} User`,
            role: 'user'
        };
        
        localStorage.setItem('currentUser', JSON.stringify(mockSocialUser));
        window.location.href = 'dashboard.html';
    }, 2000);
}

function showMessage(message, type) {
    // Remove existing messages
    const existingMessage = document.querySelector('.message');
    if (existingMessage) {
        existingMessage.remove();
    }
    
    // Create message element
    const messageEl = document.createElement('div');
    messageEl.className = `message ${type}`;
    messageEl.textContent = message;
    
    // Style the message
    messageEl.style.cssText = `
        position: fixed;
        top: 100px;
        left: 50%;
        transform: translateX(-50%);
        padding: 1rem 2rem;
        border-radius: 10px;
        color: white;
        font-weight: 500;
        z-index: 1000;
        animation: slideDown 0.3s ease;
        ${type === 'success' ? 'background: #10b981;' : ''}
        ${type === 'error' ? 'background: #ef4444;' : ''}
        ${type === 'info' ? 'background: #3b82f6;' : ''}
    `;
    
    document.body.appendChild(messageEl);
    
    // Remove message after 3 seconds
    setTimeout(() => {
        messageEl.style.animation = 'slideUp 0.3s ease';
        setTimeout(() => messageEl.remove(), 300);
    }, 3000);
}

// Add CSS for message animations
const style = document.createElement('style');
style.textContent = `
    @keyframes slideDown {
        from { transform: translateX(-50%) translateY(-20px); opacity: 0; }
        to { transform: translateX(-50%) translateY(0); opacity: 1; }
    }
    
    @keyframes slideUp {
        from { transform: translateX(-50%) translateY(0); opacity: 1; }
        to { transform: translateX(-50%) translateY(-20px); opacity: 0; }
    }
`;
document.head.appendChild(style);