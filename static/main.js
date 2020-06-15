const slides = document.querySelectorAll('.slide')
const next = document.querySelector('#next')
const prev = document.querySelector('#prev')
const auto = true
const intervalTime = 8000
let slideInterval;

const nextSlide = () => {
    // get current class
    const current = document.querySelector('.current');
    // remove the current class
    current.classList.remove('current');
    // check for next slide
    if (current.nextElementSibling) {
        //add current to next sibling
        current.nextElementSibling.classList.add('current');
    } else {
        // go back to beging if at end of list
        slides[0].classList.add('current');
    }
    //remove current class again
    setTimeout(() => current.classList.remove('current'));
};

const prevSlide = () => {
    // get current class
    const current = document.querySelector('.current')
    // remove the current class
    current.classList.remove('current')
    // check for prev slide
    if(current.previousElementSibling) {
        //add current to prev sibling
        current.previousElementSibling.classList.add('current');
    } else {
        // go to end if at begining of list
        slides[slides.length - 1].classList.add('current');
    }
    //remove current class again
    setTimeout(() => current.classList.remove('current'));
};

//button events
next.addEventListener('click', e => {
    nextSlide();
    if(auto) {
        clearInterval(slideInterval);
        slideInterval = setInterval(nextSlide, intervalTime)
    }
});

prev.addEventListener('click', e => {
    prevSlide();
});

//auto slide
if(auto) {
    slideInterval = setInterval(nextSlide, intervalTime);
}
