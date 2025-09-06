// Check if user is logged in
document.addEventListener('DOMContentLoaded', function() {
    const currentUser = JSON.parse(localStorage.getItem('currentUser'));
    
    if (!currentUser) {
        window.location.href = 'login.html';
        return;
    }
    
    // Update user name in navbar
    document.getElementById('userName').textContent = currentUser.name;
    
    // Show admin features if user is admin
    if (currentUser.role === 'admin') {
        document.getElementById('adminLink').style.display = 'block';
        addAdminFeatures();
    }
});

function showSection(sectionId) {
    // Hide all sections
    document.querySelectorAll('.content-section').forEach(section => {
        section.classList.remove('active');
    });
    
    // Remove active class from all menu items
    document.querySelectorAll('.menu-item').forEach(item => {
        item.classList.remove('active');
    });
    
    // Show selected section
    document.getElementById(sectionId).classList.add('active');
    
    // Add active class to clicked menu item
    event.target.classList.add('active');
}

function logout() {
    localStorage.removeItem('currentUser');
    window.location.href = 'index.html';
}

function startCamera() {
    window.location.href = 'recognition.html';
}

function uploadPhoto() {
    // Create file input
    const input = document.createElement('input');
    input.type = 'file';
    input.accept = 'image/*';
    
    input.onchange = function(event) {
        const file = event.target.files[0];
        if (file) {
            showMessage(`Processing ${file.name}...`, 'info');
            // This would send the file to the face recognition backend
            
            setTimeout(() => {
                showMessage('Face recognition completed!', 'success');
                updateHistory();
            }, 2000);
        }
    };
    
    input.click();
}

function updateHistory() {
    // Add new recognition to history
    const historyList = document.querySelector('.history-list');
    const newItem = document.createElement('div');
    newItem.className = 'history-item';
    newItem.innerHTML = `
        <div class="history-info">
            <h4>Recognition #${Date.now()}</h4>
            <p>Success • Just now</p>
        </div>
        <span class="status success">✓</span>
    `;
    
    historyList.insertBefore(newItem, historyList.firstChild);
    
    // Update stats
    const totalRecognitions = document.querySelector('.stat-number');
    const currentCount = parseInt(totalRecognitions.textContent);
    totalRecognitions.textContent = currentCount + 1;
}

function addAdminFeatures() {
    // Add admin-specific menu items
    const sidebar = document.querySelector('.sidebar-menu');
    const adminItem = document.createElement('a');
    adminItem.href = '#';
    adminItem.className = 'menu-item';
    adminItem.innerHTML = '<span>⚙️</span> Admin Panel';
    adminItem.onclick = () => showSection('admin');
    
    sidebar.appendChild(adminItem);
    
    // Add admin section
    const content = document.querySelector('.dashboard-content');
    const adminSection = document.createElement('div');
    adminSection.id = 'admin';
    adminSection.className = 'content-section';
    adminSection.innerHTML = `
        <h1>Admin Panel</h1>
        <div class="admin-stats">
            <div class="stat-card">
                <h3>Total Users</h3>
                <p class="stat-number">1,234</p>
            </div>
            <div class="stat-card">
                <h3>System Uptime</h3>
                <p class="stat-number">99.9%</p>
            </div>
            <div class="stat-card">
                <h3>Daily Recognitions</h3>
                <p class="stat-number">5,678</p>
            </div>
        </div>
        <div class="admin-actions">
            <button class="btn-primary" onclick="manageUsers()">Manage Users</button>
            <button class="btn-outline" onclick="viewLogs()">View System Logs</button>
        </div>
    `;
    
    content.appendChild(adminSection);
}

function manageUsers() {
    showMessage('User management panel would open here', 'info');
}

function viewLogs() {
    showMessage('System logs would be displayed here', 'info');
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