/* =========================================
SMOOTH SCROLL NAVIGATION
========================================= */
document.querySelectorAll('.sidebar a').forEach(anchor => {
anchor.addEventListener('click', function(e) {
e.preventDefault();

```
const targetId = this.getAttribute('href');
const target = document.querySelector(targetId);

if (target) {
  target.scrollIntoView({
    behavior: 'smooth',
    block: 'start'
  });
}
```

});
});

/* =========================================
ACTIVE LINK ON CLICK
========================================= */
const links = document.querySelectorAll('.sidebar a');

links.forEach(link => {
link.addEventListener('click', function() {
links.forEach(l => l.classList.remove('active'));
this.classList.add('active');
});
});

/* =========================================
ACTIVE LINK ON SCROLL
========================================= */
const sections = document.querySelectorAll('section');

window.addEventListener('scroll', () => {
let current = "";

sections.forEach(section => {
const sectionTop = section.offsetTop - 100;
const sectionHeight = section.clientHeight;

```
if (scrollY >= sectionTop && scrollY < sectionTop + sectionHeight) {
  current = section.getAttribute("id");
}
```

});

links.forEach(link => {
link.classList.remove("active");
if (link.getAttribute("href") === "#" + current) {
link.classList.add("active");
}
});
});

/* =========================================
SIMPLE FADE-IN ANIMATION
========================================= */
const revealElements = document.querySelectorAll('.card, .hero-content, .section h2');

const observer = new IntersectionObserver((entries) => {
entries.forEach(entry => {
if (entry.isIntersecting) {
entry.target.style.opacity = 1;
entry.target.style.transform = "translateY(0)";
}
});
}, {
threshold: 0.2
});

revealElements.forEach(el => {
el.style.opacity = 0;
el.style.transform = "translateY(20px)";
el.style.transition = "all 0.6s ease";
observer.observe(el);
});
