const messages = document.getElementById('messages');

function sendMessage() {
    const userInput = document.getElementById('userInput');
    const userMessage = userInput.value;

    if (userMessage) {
        // Display user's message
        messages.innerHTML += `<div>User: ${userMessage}</div>`;
        userInput.value = '';

        // Bot response
        setTimeout(() => {
            let botResponse = '';
            if (userMessage.toLowerCase().includes('about you')) {
                botResponse = "I'm Akthar Pasha, a Computer Science Engineering student passionate about web development.";
            } else if (userMessage.toLowerCase().includes('skills')) {
                botResponse = "I have skills in C, Python, JavaScript, HTML, CSS, and frameworks like React and Bootstrap.";
            } else {
                botResponse = "I'm here to answer any questions about me!";
            }
            messages.innerHTML += `<div>Bot: ${botResponse}</div>`;
            messages.scrollTop = messages.scrollHeight; // Scroll to the bottom
        }, 1000);
    }
}
