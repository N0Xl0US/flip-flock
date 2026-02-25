import countCellsWGSL from './shaders/countCells.wgsl?raw';
import scatterWGSL from './shaders/scatter.wgsl?raw';
import updateBoidsWGSL from './shaders/updateBoids.wgsl?raw';
import spriteWGSL from './shaders/sprite.wgsl?raw';

export class Simulation {
    constructor() {
        this.device = null;
        this.canvas = null;
        this.t = 0;
        this._lastTime = 0;
        this._frameCount = 0;
        this._fps = 0;
        this.onFpsUpdate = null;
        this.NUM_BOIDS = 25000;
        this.MAX_BOIDS = 500000;
        this.WORKGROUP_SIZE = 256;

        this.simParams = {
            deltaT: 1.0,
            visualRange: 45,
            cohesion: 0.0005,
            separation: 0.05,
            alignment: 0.02,
            separationDist: 12,
            maxSpeed: 4.0,
            minSpeed: 2.0,
            turnMargin: 150,
            turnFactor: 0.1,
        };
    }

    async init(canvas) {
        this.canvas = canvas;
        if (!navigator.gpu) throw new Error('WebGPU is not supported.');
        const adapter = await navigator.gpu.requestAdapter();
        if (!adapter) throw new Error('No GPU adapter found.');

        this.device = await adapter.requestDevice({
            requiredLimits: {
                maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize,
                maxBufferSize: adapter.limits.maxBufferSize,
            }
        });

        this.device.addEventListener('uncapturederror', (e) => {
            console.error('WebGPU error:', e.error.message);
        });

        this.context = canvas.getContext('webgpu');
        this.presentationFormat = navigator.gpu.getPreferredCanvasFormat();
        this.context.configure({
            device: this.device,
            format: this.presentationFormat,
            alphaMode: 'opaque',
        });

        this._resizeCanvas();
        this._setupGrid();
        this._createBuffers();
        this._createPipelines();
    }

    _resizeCanvas() {
        const dpr = window.devicePixelRatio || 1;
        this.canvas.width = this.canvas.clientWidth * dpr;
        this.canvas.height = this.canvas.clientHeight * dpr;
        this.boundsX = this.canvas.width;
        this.boundsY = this.canvas.height;
    }

    _setupGrid() {
        this.cellSize = this.simParams.visualRange;
        this.gridSizeX = Math.ceil(this.boundsX / this.cellSize);
        this.gridSizeY = Math.ceil(this.boundsY / this.cellSize);
        this.numCells = this.gridSizeX * this.gridSizeY;
    }

    _createBuffers() {
        const N = this.MAX_BOIDS;

        const posData = new Float32Array(N * 2);
        const velData = new Float32Array(N * 2);
        for (let i = 0; i < N; i++) {
            posData[i * 2] = Math.random() * this.boundsX;
            posData[i * 2 + 1] = Math.random() * this.boundsY;
            const angle = Math.random() * Math.PI * 2;
            const speed = this.simParams.minSpeed +
                Math.random() * (this.simParams.maxSpeed - this.simParams.minSpeed);
            velData[i * 2] = Math.cos(angle) * speed;
            velData[i * 2 + 1] = Math.sin(angle) * speed;
        }

        const mkInit = (data) => {
            const b = this.device.createBuffer({
                size: data.byteLength,
                usage: GPUBufferUsage.STORAGE | GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
                mappedAtCreation: true,
            });
            new Float32Array(b.getMappedRange()).set(data);
            b.unmap();
            return b;
        };

        this.posBuffers = [mkInit(posData), mkInit(posData)];
        this.velBuffers = [mkInit(velData), mkInit(velData)];

        this.cellCountsBuf = this.device.createBuffer({
            size: this.numCells * 4,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
        });
        this.cellOffsetsBuf = this.device.createBuffer({
            size: this.numCells * 4,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });
        this.cellCountsReadBuf = this.device.createBuffer({
            size: this.numCells * 4,
            usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
        });
        this.sortedIndicesBuf = this.device.createBuffer({
            size: N * 4,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });
        this.scatterOffsetsBuf = this.device.createBuffer({
            size: this.numCells * 4,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });

        this.gridParamsBuf = this.device.createBuffer({
            size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });
        this.simParamsBuf = this.device.createBuffer({
            size: 64, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });
        this.renderParamsBuf = this.device.createBuffer({
            size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });

        this._zeroCounts = new Uint32Array(this.numCells);
    }

    _createPipelines() {
        const cs = (code) => this.device.createComputePipeline({
            layout: 'auto',
            compute: { module: this.device.createShaderModule({ code }), entryPoint: 'main' },
        });

        this.countCellsPL = cs(countCellsWGSL);
        this.scatterPL = cs(scatterWGSL);
        this.updateBoidsPL = cs(updateBoidsWGSL);

        const spriteModule = this.device.createShaderModule({ code: spriteWGSL });
        this.renderPL = this.device.createRenderPipeline({
            layout: 'auto',
            vertex: {
                module: spriteModule,
                entryPoint: 'vert_main',
                buffers: [{
                    arrayStride: 8,
                    stepMode: 'instance',
                    attributes: [{ shaderLocation: 0, offset: 0, format: 'float32x2' }],
                }, {
                    arrayStride: 8,
                    stepMode: 'instance',
                    attributes: [{ shaderLocation: 1, offset: 0, format: 'float32x2' }],
                }],
            },
            fragment: {
                module: spriteModule,
                entryPoint: 'frag_main',
                targets: [{ format: this.presentationFormat }],
            },
            primitive: { topology: 'triangle-list' },
        });
    }

    updateParams(params) {
        Object.assign(this.simParams, params);
    }

    resize() {
        this._resizeCanvas();
        const old = this.numCells;
        this._setupGrid();
        if (this.numCells !== old) {
            this.cellCountsBuf.destroy();
            this.cellOffsetsBuf.destroy();
            this.cellCountsReadBuf.destroy();
            this.scatterOffsetsBuf.destroy();
            this.cellCountsBuf = this.device.createBuffer({
                size: this.numCells * 4,
                usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
            });
            this.cellOffsetsBuf = this.device.createBuffer({
                size: this.numCells * 4,
                usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
            });
            this.cellCountsReadBuf = this.device.createBuffer({
                size: this.numCells * 4,
                usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
            });
            this.scatterOffsetsBuf = this.device.createBuffer({
                size: this.numCells * 4,
                usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
            });
            this._zeroCounts = new Uint32Array(this.numCells);
        }
    }

    async frame(timestamp) {
        this._frameCount++;
        if (timestamp - this._lastTime >= 1000) {
            this._fps = this._frameCount;
            this._frameCount = 0;
            this._lastTime = timestamp;
            if (this.onFpsUpdate) this.onFpsUpdate(this._fps);
        }

        const src = this.t % 2;
        const dst = (this.t + 1) % 2;
        const boidWg = Math.ceil(this.NUM_BOIDS / this.WORKGROUP_SIZE);

        this._writeUniforms();

        // Clear cell counts
        this.device.queue.writeBuffer(this.cellCountsBuf, 0, this._zeroCounts);

        // 1. Count boids per cell
        const enc1 = this.device.createCommandEncoder();
        {
            const bg = this.device.createBindGroup({
                layout: this.countCellsPL.getBindGroupLayout(0),
                entries: [
                    { binding: 0, resource: { buffer: this.gridParamsBuf } },
                    { binding: 1, resource: { buffer: this.posBuffers[src] } },
                    { binding: 2, resource: { buffer: this.cellCountsBuf } },
                ],
            });
            const pass = enc1.beginComputePass();
            pass.setPipeline(this.countCellsPL);
            pass.setBindGroup(0, bg);
            pass.dispatchWorkgroups(boidWg);
            pass.end();
        }
        enc1.copyBufferToBuffer(this.cellCountsBuf, 0, this.cellCountsReadBuf, 0, this.numCells * 4);
        this.device.queue.submit([enc1.finish()]);

        // 2. CPU prefix sum (grid is tiny: ~1500 cells = instant)
        await this.cellCountsReadBuf.mapAsync(GPUMapMode.READ);
        const rawCounts = new Uint32Array(this.cellCountsReadBuf.getMappedRange());
        const counts = rawCounts.slice();
        this.cellCountsReadBuf.unmap();

        const offsets = new Uint32Array(this.numCells);
        let sum = 0;
        for (let i = 0; i < this.numCells; i++) {
            offsets[i] = sum;
            sum += counts[i];
        }

        this.device.queue.writeBuffer(this.cellOffsetsBuf, 0, offsets);
        this.device.queue.writeBuffer(this.cellCountsBuf, 0, counts);
        this.device.queue.writeBuffer(this.scatterOffsetsBuf, 0, offsets.slice());

        // 3. Scatter + Update + Render in one submission
        const enc2 = this.device.createCommandEncoder();

        // Scatter boids to sorted positions
        {
            const bg = this.device.createBindGroup({
                layout: this.scatterPL.getBindGroupLayout(0),
                entries: [
                    { binding: 0, resource: { buffer: this.gridParamsBuf } },
                    { binding: 1, resource: { buffer: this.posBuffers[src] } },
                    { binding: 2, resource: { buffer: this.scatterOffsetsBuf } },
                    { binding: 3, resource: { buffer: this.sortedIndicesBuf } },
                ],
            });
            const pass = enc2.beginComputePass();
            pass.setPipeline(this.scatterPL);
            pass.setBindGroup(0, bg);
            pass.dispatchWorkgroups(boidWg);
            pass.end();
        }

        // Update boids
        {
            const bg = this.device.createBindGroup({
                layout: this.updateBoidsPL.getBindGroupLayout(0),
                entries: [
                    { binding: 0, resource: { buffer: this.simParamsBuf } },
                    { binding: 1, resource: { buffer: this.posBuffers[src] } },
                    { binding: 2, resource: { buffer: this.velBuffers[src] } },
                    { binding: 3, resource: { buffer: this.posBuffers[dst] } },
                    { binding: 4, resource: { buffer: this.velBuffers[dst] } },
                    { binding: 5, resource: { buffer: this.sortedIndicesBuf } },
                    { binding: 6, resource: { buffer: this.cellOffsetsBuf } },
                    { binding: 7, resource: { buffer: this.cellCountsBuf } },
                ],
            });
            const pass = enc2.beginComputePass();
            pass.setPipeline(this.updateBoidsPL);
            pass.setBindGroup(0, bg);
            pass.dispatchWorkgroups(boidWg);
            pass.end();
        }

        // Render
        {
            const textureView = this.context.getCurrentTexture().createView();
            const renderBG = this.device.createBindGroup({
                layout: this.renderPL.getBindGroupLayout(0),
                entries: [{ binding: 0, resource: { buffer: this.renderParamsBuf } }],
            });
            const pass = enc2.beginRenderPass({
                colorAttachments: [{
                    view: textureView,
                    clearValue: { r: 0.16, g: 0.16, b: 0.21, a: 1.0 }, // Dark blue-grey background
                    loadOp: 'clear',
                    storeOp: 'store',
                }],
            });
            pass.setPipeline(this.renderPL);
            pass.setBindGroup(0, renderBG);
            pass.setVertexBuffer(0, this.posBuffers[dst]);
            pass.setVertexBuffer(1, this.velBuffers[dst]);
            pass.draw(3, this.NUM_BOIDS, 0, 0);
            pass.end();
        }

        this.device.queue.submit([enc2.finish()]);
        this.t++;
    }

    _writeUniforms() {
        const p = this.simParams;

        const gp = new ArrayBuffer(16);
        new Uint32Array(gp, 0, 2).set([this.gridSizeX, this.gridSizeY]);
        new Float32Array(gp, 8, 1).set([this.cellSize]);
        new Uint32Array(gp, 12, 1).set([this.NUM_BOIDS]);
        this.device.queue.writeBuffer(this.gridParamsBuf, 0, gp);

        const sb = new ArrayBuffer(64);
        const sf = new Float32Array(sb);
        const su = new Uint32Array(sb);
        sf[0] = p.deltaT;
        sf[1] = p.visualRange;
        sf[2] = p.cohesion;
        sf[3] = p.separation;
        sf[4] = p.alignment;
        sf[5] = p.separationDist;
        sf[6] = p.maxSpeed;
        sf[7] = p.minSpeed;
        sf[8] = this.boundsX;
        sf[9] = this.boundsY;
        sf[10] = p.turnMargin;
        sf[11] = p.turnFactor;
        su[12] = this.NUM_BOIDS;
        su[13] = this.gridSizeX;
        su[14] = this.gridSizeY;
        sf[15] = this.cellSize;
        this.device.queue.writeBuffer(this.simParamsBuf, 0, sb);

        // Extremely small boid size to look like dots (as in the target image)
        const boidSize = Math.max(0.2, 0.15 * (1000 / Math.sqrt(this.NUM_BOIDS)));
        this.device.queue.writeBuffer(this.renderParamsBuf, 0,
            new Float32Array([this.boundsX, this.boundsY, boidSize, 0]));
    }
}
