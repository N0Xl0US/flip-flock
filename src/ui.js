import GUI from 'lil-gui';

export function createUI(simulation) {
    const gui = new GUI({ title: 'Flip Flock' });

    const fpsObj = { FPS: '0' };
    gui.add(fpsObj, 'FPS').listen().disable();
    simulation.onFpsUpdate = (fps) => { fpsObj.FPS = String(fps); };

    const p = simulation.simParams;
    const proxy = {
        get Boids() { return simulation.NUM_BOIDS; },
        set Boids(v) { simulation.NUM_BOIDS = Math.floor(v); },
        get Cohesion() { return (p.cohesion / 0.002) * 100; },
        set Cohesion(v) { p.cohesion = (v / 100) * 0.002; simulation.updateParams({}); },
        get Separation() { return (p.separation / 0.2) * 100; },
        set Separation(v) { p.separation = (v / 100) * 0.2; simulation.updateParams({}); },
        get Alignment() { return (p.alignment / 0.08) * 100; },
        set Alignment(v) { p.alignment = (v / 100) * 0.08; simulation.updateParams({}); },
        get Speed() { return (p.maxSpeed / 8.0) * 100; },
        set Speed(v) { p.maxSpeed = (v / 100) * 8.0; simulation.updateParams({}); },
    };

    gui.add(proxy, 'Boids', 100, simulation.MAX_BOIDS, 100).name('Boid Count');
    gui.add(proxy, 'Cohesion', 0, 100, 1);
    gui.add(proxy, 'Separation', 0, 100, 1);
    gui.add(proxy, 'Alignment', 0, 100, 1);
    gui.add(proxy, 'Speed', 0, 100, 1);

    return gui;
}
