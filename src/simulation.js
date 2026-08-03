import countCellsWGSL from './shaders/countCells.wgsl?raw';
import scatterWGSL from './shaders/scatter.wgsl?raw';
import updateBoidsWGSL from './shaders/updateBoids.wgsl?raw';
import spriteWGSL from './shaders/sprite.wgsl?raw';
import prefixSumWGSL from './shaders/prefixSum.wgsl?raw';
import clearBufferWGSL from './shaders/clearBuffer.wgsl?raw';

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
        this.MAX_BOIDS = 50000;
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
        this._createBindGroups();
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
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });
        this.cellOffsetsBuf = this.device.createBuffer({
            size: this.numCells * 4,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
        });
        this.sortedIndicesBuf = this.device.createBuffer({
            size: N * 4,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });
        // Scatter uses atomic offsets — separate buffer that starts as a copy of cellOffsets
        this.scatterOffsetsBuf = this.device.createBuffer({
            size: this.numCells * 4,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
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

        // Prefix sum params buffer
        this.prefixSumParamsBuf = this.device.createBuffer({
            size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });

        // Clear params buffer
        this.clearParamsBuf = this.device.createBuffer({
            size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });

        // Pre-allocate reusable typed arrays for uniform writes
        this._gridParamsAB = new ArrayBuffer(16);
        this._simParamsAB = new ArrayBuffer(64);
        this._renderParamsF32 = new Float32Array(4);
    }

    _createPipelines() {
        const cs = (code) => this.device.createComputePipeline({
            layout: 'auto',
            compute: { module: this.device.createShaderModule({ code }), entryPoint: 'main' },
        });

        this.countCellsPL = cs(countCellsWGSL);
        this.scatterPL = cs(scatterWGSL);
        this.updateBoidsPL = cs(updateBoidsWGSL);
        this.prefixSumPL = cs(prefixSumWGSL);
        this.clearBufferPL = cs(clearBufferWGSL);

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

    _createBindGroups() {
        // Clear cell counts bind group
        this.clearCellCountsBG = this.device.createBindGroup({
            layout: this.clearBufferPL.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: { buffer: this.cellCountsBuf } },
                { binding: 1, resource: { buffer: this.clearParamsBuf } },
            ],
        });

        // Count cells bind groups (one per ping-pong source)
        this.countCellsBGs = [0, 1].map(src => this.device.createBindGroup({
            layout: this.countCellsPL.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: { buffer: this.gridParamsBuf } },
                { binding: 1, resource: { buffer: this.posBuffers[src] } },
                { binding: 2, resource: { buffer: this.cellCountsBuf } },
            ],
        }));

        // GPU prefix sum bind group
        this.prefixSumBG = this.device.createBindGroup({
            layout: this.prefixSumPL.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: { buffer: this.prefixSumParamsBuf } },
                { binding: 1, resource: { buffer: this.cellCountsBuf } },
                { binding: 2, resource: { buffer: this.cellOffsetsBuf } },
            ],
        });

        // Scatter bind groups (one per ping-pong source)
        this.scatterBGs = [0, 1].map(src => this.device.createBindGroup({
            layout: this.scatterPL.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: { buffer: this.gridParamsBuf } },
                { binding: 1, resource: { buffer: this.posBuffers[src] } },
                { binding: 2, resource: { buffer: this.scatterOffsetsBuf } },
                { binding: 3, resource: { buffer: this.sortedIndicesBuf } },
            ],
        }));

        // Update boids bind groups (one per ping-pong direction)
        this.updateBoidsBGs = [0, 1].map(src => {
            const dst = 1 - src;
            return this.device.createBindGroup({
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
        });

        // Render bind group (shared — doesn't depend on ping-pong)
        this.renderBG = this.device.createBindGroup({
            layout: this.renderPL.getBindGroupLayout(0),
            entries: [{ binding: 0, resource: { buffer: this.renderParamsBuf } }],
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
            this.scatterOffsetsBuf.destroy();
            this.cellCountsBuf = this.device.createBuffer({
                size: this.numCells * 4,
                usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
            });
            this.cellOffsetsBuf = this.device.createBuffer({
                size: this.numCells * 4,
                usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
            });
            this.scatterOffsetsBuf = this.device.createBuffer({
                size: this.numCells * 4,
                usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
            });
            // Rebuild all bind groups that reference grid buffers
            this._createBindGroups();
        }
    }

    frame(timestamp) {
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
        const cellWg = Math.ceil(this.numCells / this.WORKGROUP_SIZE);

        this._writeUniforms();

        // Single command encoder for the entire frame — no GPU↔CPU sync!
        const enc = this.device.createCommandEncoder();

        // 1. Clear cell counts on GPU
        {
            const pass = enc.beginComputePass();
            pass.setPipeline(this.clearBufferPL);
            pass.setBindGroup(0, this.clearCellCountsBG);
            pass.dispatchWorkgroups(cellWg);
            pass.end();
        }

        // 2. Count boids per cell
        {
            const pass = enc.beginComputePass();
            pass.setPipeline(this.countCellsPL);
            pass.setBindGroup(0, this.countCellsBGs[src]);
            pass.dispatchWorkgroups(boidWg);
            pass.end();
        }

        // 3. GPU prefix sum (replaces the CPU mapAsync bottleneck)
        {
            const pass = enc.beginComputePass();
            pass.setPipeline(this.prefixSumPL);
            pass.setBindGroup(0, this.prefixSumBG);
            pass.dispatchWorkgroups(1);
            pass.end();
        }

        // 4. Copy offsets for scatter (scatter uses atomics so needs its own copy)
        enc.copyBufferToBuffer(this.cellOffsetsBuf, 0, this.scatterOffsetsBuf, 0, this.numCells * 4);

        // 5. Scatter boids to sorted positions
        {
            const pass = enc.beginComputePass();
            pass.setPipeline(this.scatterPL);
            pass.setBindGroup(0, this.scatterBGs[src]);
            pass.dispatchWorkgroups(boidWg);
            pass.end();
        }

        // 6. Update boids
        {
            const pass = enc.beginComputePass();
            pass.setPipeline(this.updateBoidsPL);
            pass.setBindGroup(0, this.updateBoidsBGs[src]);
            pass.dispatchWorkgroups(boidWg);
            pass.end();
        }

        // 7. Render
        {
            const textureView = this.context.getCurrentTexture().createView();
            const pass = enc.beginRenderPass({
                colorAttachments: [{
                    view: textureView,
                    clearValue: { r: 0.16, g: 0.16, b: 0.21, a: 1.0 }, // Dark blue-grey background
                    loadOp: 'clear',
                    storeOp: 'store',
                }],
            });
            pass.setPipeline(this.renderPL);
            pass.setBindGroup(0, this.renderBG);
            pass.setVertexBuffer(0, this.posBuffers[dst]);
            pass.setVertexBuffer(1, this.velBuffers[dst]);
            pass.draw(3, this.NUM_BOIDS, 0, 0);
            pass.end();
        }

        this.device.queue.submit([enc.finish()]);
        this.t++;
    }

    _writeUniforms() {
        const p = this.simParams;

        const gp = this._gridParamsAB;
        new Uint32Array(gp, 0, 2).set([this.gridSizeX, this.gridSizeY]);
        new Float32Array(gp, 8, 1).set([this.cellSize]);
        new Uint32Array(gp, 12, 1).set([this.NUM_BOIDS]);
        this.device.queue.writeBuffer(this.gridParamsBuf, 0, gp);

        const sb = this._simParamsAB;
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
        const rp = this._renderParamsF32;
        rp[0] = this.boundsX; rp[1] = this.boundsY; rp[2] = boidSize; rp[3] = 0;
        this.device.queue.writeBuffer(this.renderParamsBuf, 0, rp);

        // Prefix sum params
        this.device.queue.writeBuffer(this.prefixSumParamsBuf, 0,
            new Uint32Array([this.numCells, 0, 0, 0]));

        // Clear params (clear uses 0 fill, count = numCells)
        this.device.queue.writeBuffer(this.clearParamsBuf, 0,
            new Uint32Array([this.numCells, 0, 0, 0]));
    }
}
