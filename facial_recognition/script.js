function startRecognition() {
    const button = document.querySelector('.btn-primary');
    
    // Add pulse animation
    button.classList.add('pulse');
    
    // Simulate loading state
    const originalText = button.innerHTML;
    button.innerHTML = '<span>Initializing...</span>';
    
    setTimeout(() => {
        button.innerHTML = originalText;
        button.classList.remove('pulse');
        
        // Redirect to recognition page
        window.location.href = 'recognition.html';
    }, 1500);
}

// Add smooth scroll behavior
document.addEventListener('DOMContentLoaded', function() {
    // Animate elements on scroll
    const observerOptions = {
        threshold: 0.1,
        rootMargin: '0px 0px -50px 0px'
    };
    
    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.style.opacity = '1';
                entry.target.style.transform = 'translateY(0)';
            }
        });
    }, observerOptions);
    
    // Observe feature elements
    document.querySelectorAll('.feature').forEach(feature => {
        feature.style.opacity = '0';
        feature.style.transform = 'translateY(20px)';
        feature.style.transition = 'all 0.6s ease';
        observer.observe(feature);
    });
});

// Add keyboard navigation
document.addEventListener('keydown', function(e) {
    if (e.key === 'Enter' && e.target.classList.contains('btn-primary')) {
        startRecognition();
    }
});