document.addEventListener('DOMContentLoaded', () => {
    const buttons = document.querySelectorAll('.pay-btn');
    
    buttons.forEach(btn => {
        btn.addEventListener('click', (e) => {
            const name = e.target.getAttribute('data-product');
            const price = e.target.getAttribute('data-price');
            alert(`Initiating secure payment for: ${name}\nTotal: $${price}`);
            // Integrate Stripe or PayPal API here
        });
    });
});