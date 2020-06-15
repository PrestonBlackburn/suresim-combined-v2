
const navSlide = () => {
    const mobile = document.querySelector('.mobile');
    const nav = document.querySelector('.nav-links');
    // const slides = document.querySelectorAll('.slider')
    const navLinks = document.querySelectorAll('.nav-links li');

    mobile.addEventListener('click', () => {
        nav.classList.toggle('nav-active');
        // slides.classList.toggle('nav-active-slider')
        
    });

}

navSlide();