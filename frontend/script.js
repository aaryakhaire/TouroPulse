/* ═══════════════════════════════════════════════════════════
   TOUROPULSE — Frontend Script
   Handles: Scroll reveals, nav state, counters, parallax, chatbot
   ═══════════════════════════════════════════════════════════ */

// ── REVEAL ON SCROLL (IntersectionObserver) ──
const revealObserver = new IntersectionObserver((entries) => {
    entries.forEach(entry => {
        if (entry.isIntersecting) {
            entry.target.classList.add('active');
        }
    });
}, {
    threshold: 0.1,
    rootMargin: '0px 0px -40px 0px'
});

document.querySelectorAll('.reveal-up').forEach(el => revealObserver.observe(el));

// ── NAVBAR SCROLL STATE ──
const nav = document.getElementById('main-nav');
let lastScroll = 0;

window.addEventListener('scroll', () => {
    const scroll = window.scrollY;
    
    if (scroll > 80) {
        nav.classList.add('scrolled');
    } else {
        nav.classList.remove('scrolled');
    }
    
    lastScroll = scroll;
});

// ── HERO PARALLAX (subtle) ──
const heroImg = document.getElementById('hero-img');
if (heroImg) {
    window.addEventListener('scroll', () => {
        const scroll = window.scrollY;
        const heroHeight = window.innerHeight;
        if (scroll < heroHeight) {
            const parallax = scroll * 0.3;
            heroImg.style.transform = `scale(1.05) translateY(${parallax}px)`;
        }
    });
}

// ── COUNTER ANIMATION ──
const animateCounters = () => {
    const counters = document.querySelectorAll('.counter');
    counters.forEach(counter => {
        const target = parseInt(counter.getAttribute('data-target'));
        if (!target) return;
        
        let count = 0;
        const duration = 2000; // ms
        const increment = target / (duration / 16);
        
        const update = () => {
            count += increment;
            if (count < target) {
                counter.innerText = Math.ceil(count).toLocaleString();
                requestAnimationFrame(update);
            } else {
                counter.innerText = target.toLocaleString();
            }
        };
        
        update();
    });
};

const counterObserver = new IntersectionObserver((entries) => {
    entries.forEach(entry => {
        if (entry.isIntersecting) {
            animateCounters();
            counterObserver.unobserve(entry.target);
        }
    });
}, { threshold: 0.3 });

const statsSection = document.querySelector('.stats-section');
if (statsSection) counterObserver.observe(statsSection);

// ── HAMBURGER MENU (Mobile) ──
const hamburger = document.getElementById('nav-hamburger');
const navLinks = document.querySelector('.nav-links');

if (hamburger) {
    hamburger.addEventListener('click', () => {
        navLinks.style.display = navLinks.style.display === 'flex' ? 'none' : 'flex';
        navLinks.style.flexDirection = 'column';
        navLinks.style.position = 'absolute';
        navLinks.style.top = '70px';
        navLinks.style.right = '5%';
        navLinks.style.background = 'rgba(10,10,10,0.95)';
        navLinks.style.padding = '20px 30px';
        navLinks.style.gap = '20px';
        navLinks.style.borderRadius = '0';
        navLinks.style.border = '1px solid rgba(245,240,235,0.08)';
    });
}

// ── SMOOTH SCROLL for anchor links ──
document.querySelectorAll('a[href^="#"]').forEach(anchor => {
    anchor.addEventListener('click', function (e) {
        e.preventDefault();
        const target = document.querySelector(this.getAttribute('href'));
        if (target) {
            target.scrollIntoView({ behavior: 'smooth', block: 'start' });
        }
        // Close mobile menu if open
        if (window.innerWidth <= 768 && navLinks) {
            navLinks.style.display = 'none';
        }
    });
});

// ── CHATBOT ──
const chatToggle = document.getElementById('chat-toggle');
const chatWindow = document.getElementById('chat-window');
const chatClose = document.getElementById('chat-close');
const chatInput = document.getElementById('chat-input');
const chatHistory = document.getElementById('chat-history');
const chatSend = document.getElementById('chat-send');

if (chatToggle && chatWindow) {
    chatToggle.addEventListener('click', () => {
        chatWindow.classList.toggle('chat-window-hidden');
    });
}

if (chatClose) {
    chatClose.addEventListener('click', () => {
        chatWindow.classList.add('chat-window-hidden');
    });
}

const sendMessage = async () => {
    const message = chatInput.value.trim();
    if (!message) return;

    // Add user message
    const userDiv = document.createElement('div');
    userDiv.className = 'msg msg-user';
    userDiv.textContent = message;
    chatHistory.appendChild(userDiv);
    chatInput.value = '';
    chatHistory.scrollTop = chatHistory.scrollHeight;

    // AI thinking placeholder
    const aiDiv = document.createElement('div');
    aiDiv.className = 'msg msg-ai';
    aiDiv.textContent = 'Analyzing...';
    chatHistory.appendChild(aiDiv);
    chatHistory.scrollTop = chatHistory.scrollHeight;

    try {
        // Dynamic API Endpoint for Deployment (Viva-Ready)
        const API_BASE = window.location.hostname === '127.0.0.1' || window.location.hostname === 'localhost' 
                         ? 'http://127.0.0.1:8001' 
                         : ''; // Set empty for relative requests on same-origin deployment
        
        const response = await fetch(`${API_BASE}/chat`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ message })
        });
        const data = await response.json();
        aiDiv.textContent = data.response;
    } catch (error) {
        aiDiv.textContent = 'Connection error. Please ensure the backend is running on port 8001.';
        console.error('Chat error:', error);
    }
    chatHistory.scrollTop = chatHistory.scrollHeight;
};

if (chatSend) {
    chatSend.addEventListener('click', sendMessage);
}

if (chatInput) {
    chatInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter') sendMessage();
    });
}