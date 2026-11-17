const nav = document.querySelector('.navbar');
const toggle = document.querySelector('.menu-toggle');
const links = [...document.querySelectorAll('.nav-links a')];
function closeMenu() {
  nav.classList.remove('menu-open');
  toggle.setAttribute('aria-expanded', 'false');
}
toggle.addEventListener('click', () => {
  const open = nav.classList.toggle('menu-open');
  toggle.setAttribute('aria-expanded', String(open));
});
document.addEventListener('keydown', event => {
  if (event.key === 'Escape' && nav.classList.contains('menu-open')) {
    closeMenu();
    toggle.focus();
  }
});
links.forEach(link => link.addEventListener('click', closeMenu));
function updateNav() {
  nav.classList.toggle('scrolled', window.scrollY > 60);
  let current = null;
  for (const link of links) {
    const section = document.querySelector(link.getAttribute('href'));
    if (section && section.getBoundingClientRect().top <= 130) current = link;
  }
  for (const link of links) {
    link.classList.toggle('active', link === current);
    if (link === current) link.setAttribute('aria-current', 'location');
    else link.removeAttribute('aria-current');
  }
}
window.addEventListener('scroll', updateNav, {passive: true});
window.addEventListener('resize', () => {
  if (window.innerWidth > 767) closeMenu();
  updateNav();
});
updateNav();
