/**
 * CURIO project page
 * Scroll-reveal sections and copy buttons for code blocks.
 */

document.addEventListener('DOMContentLoaded', function () {

    // Reveal sections as they scroll into view.
    const observer = new IntersectionObserver(function (entries) {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.style.opacity = '1';
                entry.target.style.transform = 'translateY(0)';
            }
        });
    }, { threshold: 0.05, rootMargin: '0px 0px -50px 0px' });

    document.querySelectorAll('.section:not(.hero)').forEach(section => {
        section.style.opacity = '0';
        section.style.transform = 'translateY(20px)';
        section.style.transition = 'opacity 0.6s ease-out, transform 0.6s ease-out';
        observer.observe(section);
    });

    // Copy buttons on BibTeX and quick-start code blocks.
    document.querySelectorAll('#BibTeX pre, .code-card pre').forEach(pre => {
        const button = document.createElement('button');
        button.className = 'button is-small is-dark copy-button';
        button.innerHTML = '<span class="icon"><i class="fas fa-copy"></i></span>';
        button.title = 'Copy to clipboard';

        const container = pre.parentElement;
        container.style.position = 'relative';
        container.insertBefore(button, pre);

        button.addEventListener('click', function () {
            const code = pre.querySelector('code').textContent;
            navigator.clipboard.writeText(code).then(() => {
                button.innerHTML = '<span class="icon"><i class="fas fa-check"></i></span>';
                setTimeout(() => {
                    button.innerHTML = '<span class="icon"><i class="fas fa-copy"></i></span>';
                }, 2000);
            }).catch(err => console.error('Failed to copy:', err));
        });
    });

    document.querySelectorAll('a[target="_blank"]').forEach(link => {
        link.setAttribute('rel', 'noopener noreferrer');
    });
});
