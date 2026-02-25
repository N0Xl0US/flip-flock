import './style.css';
import { Simulation } from './simulation.js';
import { createUI } from './ui.js';

async function main() {
    const canvas = document.getElementById('canvas');
    const errorOverlay = document.getElementById('error-overlay');

    const simulation = new Simulation();

    try {
        await simulation.init(canvas);
    } catch (err) {
        console.error(err);
        errorOverlay.classList.add('visible');
        errorOverlay.textContent = err.message;
        return;
    }

    createUI(simulation);
    window.addEventListener('resize', () => simulation.resize());

    async function loop(timestamp) {
        await simulation.frame(timestamp);
        requestAnimationFrame(loop);
    }

    requestAnimationFrame(loop);
}

main();
