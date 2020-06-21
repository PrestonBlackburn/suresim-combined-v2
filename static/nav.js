
var hamburger = document.getElementById('hamburger');
var navUL = document.getElementById('nav-links');

hamburger.addEventListener('click', () => {
    navUL.classList.toggle('show');
});
