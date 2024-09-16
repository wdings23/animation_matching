const UINT32_MAX: u32 = 0xffffffffu;
const FLT_MAX: f32 = 1.0e+10;
const PI: f32 = 3.14159f;
const PROBE_IMAGE_SIZE: u32 = 8u;
const VALIDATION_STEP: u32 = 16u;

struct RandomResult {
    mfNum: f32,
    miSeed: u32,
};

struct Ray
{
    mOrigin: vec4<f32>,
    mDirection: vec4<f32>,
    mfT: vec4<f32>,
};

struct BVHNode
{
    mBoundingBoxMin : vec4<f32>,
    mBoundingBoxMax : vec4<f32>,
    miLeftFirst : u32,
    miNumTriangles : u32,
    miLevel : u32,
    mPadding : u32
};

struct Tri
{
    miV0 : u32,
    miV1 : u32,
    miV2 : u32,
    mPadding : u32,

    mCentroid : vec4<f32>
};

struct VertexInput {
    @location(0) pos : vec4<f32>,
    @location(1) texCoord: vec2<f32>,
    @location(2) normal : vec4<f32>
};
struct VertexOutput {
    @location(0) texCoord: vec2<f32>,
    @builtin(position) pos: vec4<f32>,
    @location(1) normal: vec4<f32>
};
struct FragmentOutput {
    @location(0) radianceOutput : vec4<f32>,
    @location(1) temporalReservoir: vec4<f32>,
    @location(2) ambientOcclusionOutput: vec4<f32>,
    @location(3) hitPosition: vec4<f32>,
    @location(4) hitNormal: vec4<f32>,
    @location(5) sampleRayHitPosition: vec4<f32>,
    @location(6) sampleRayDirection: vec4<f32>,
    @location(7) rayDirection: vec4<f32>,
};

struct UniformData
{
    miFrame: u32,
    miRand0: u32,
    miRand1: u32,
    miRand2: u32,

    mViewProjectionMatrix: mat4x4<f32>,
    
    miScreenWidth: u32,
    miScreenHeight: u32,
    miFrame: u32,
    miNumMaterials: u32,

    mLightDirection: vec4<f32>,
    mLightRadiance: vec4<f32>,

    mfAmbientOcclusionDistanceThreshold: f32,
};

struct TemporalRestirResult
{
    mRadiance: vec4<f32>,
    mReservoir: vec4<f32>,
    mRayDirection: vec4<f32>,
    mAmbientOcclusion: vec4<f32>,
    mHitPosition: vec4<f32>,
    mHitNormal: vec4<f32>,
    mDirectSunLight: vec4<f32>,
    mIntersectionResult: IntersectBVHResult,
    mRandomResult: RandomResult,
    mfNumValidSamples: f32,
};

struct ReservoirResult
{
    mReservoir: vec4<f32>,
    mbExchanged: bool
};

struct IntersectBVHResult
{
    mHitPosition: vec3<f32>,
    mHitNormal: vec3<f32>,
    miHitTriangle: u32,
    mBarycentricCoordinate: vec3<f32>,
};

struct RayTriangleIntersectionResult
{
    mIntersectPosition: vec3<f32>,
    mIntersectNormal: vec3<f32>,
    mBarycentricCoordinate: vec3<f32>,
};

struct DisocclusionResult
{
    mBackProjectScreenCoord: vec2<i32>,
    mbDisoccluded: bool,
};

struct DirectSunLightResult
{
    mReservoir: vec4<f32>,
    mRadiance: vec3<f32>,
    mRandomResult: RandomResult,
};

struct IrradianceCacheEntry
{
    mPosition:              vec4<f32>,
    mImageProbe:            array<vec4<f32>, 64>,            // 8x8 image probe
};

struct DefaultUniformData
{
    miScreenWidth: i32,
    miScreenHeight: i32,
    miFrame: i32,
    miNumMeshes: u32,

    mfRand0: f32,
    mfRand1: f32,
    mfRand2: f32,
    mfRand3: f32,

    mViewProjectionMatrix: mat4x4<f32>,
    mPrevViewProjectionMatrix: mat4x4<f32>,
    mViewMatrix: mat4x4<f32>,
    mProjectionMatrix: mat4x4<f32>,

    mJitteredViewProjectionMatrix: mat4x4<f32>,
    mPrevJitteredViewProjectionMatrix: mat4x4<f32>,

    mCameraPosition: vec4<f32>,
    mCameraLookDir: vec4<f32>,

    mLightRadiance: vec4<f32>,
    mLightDirection: vec4<f32>,

    mfAmbientOcclusionDistanceThreshold: f32,
};

@group(0) @binding(0)
var worldPositionTexture: texture_2d<f32>;

@group(0) @binding(1)
var normalTexture: texture_2d<f32>;

@group(0) @binding(2)
var texCoordTexture: texture_2d<f32>;

@group(0) @binding(3)
var skyTexture: texture_2d<f32>;

@group(0) @binding(4)
var prevTemporalReservoirTexture: texture_2d<f32>;

@group(0) @binding(5)
var prevTemporalRadianceTexture: texture_2d<f32>;

@group(0) @binding(6)
var prevAmbientOcclusionTexture: texture_2d<f32>;

@group(0) @binding(7)
var prevTemporalHitPositionTexture: texture_2d<f32>;

@group(0) @binding(8)
var prevTemporalHitNormalTexture: texture_2d<f32>;

@group(0) @binding(9)
var prevWorldPositionTexture: texture_2d<f32>;

@group(0) @binding(10)
var prevNormalTexture: texture_2d<f32>;

@group(0) @binding(11)
var motionVectorTexture: texture_2d<f32>;

@group(0) @binding(12)
var prevMotionVectorTexture: texture_2d<f32>;

@group(0) @binding(13)
var<storage, read> irradianceCache: array<IrradianceCacheEntry>;

@group(0) @binding(14)
var prevTemporalRayDirectionTexture: texture_2d<f32>;

@group(0) @binding(15)
var prevSpatialRadianceTexture: texture_2d<f32>;

@group(0) @binding(16)
var prevSpatialReservoirTexture: texture_2d<f32>;

@group(0) @binding(17)
var prevSpatialHitPositionTexture: texture_2d<f32>;

@group(0) @binding(18)
var prevSpatialHitNormalTexture: texture_2d<f32>;



@group(0) @binding(19)
var textureSampler: sampler;

@group(1) @binding(0)
var<uniform> uniformData: UniformData;

struct BVHNode2
{
    mMinBound: vec4<f32>,
    mMaxBound: vec4<f32>,
    mCentroid: vec4<f32>,
    
    miChildren0: u32,
    miChildren1: u32,
    miPrimitiveID: u32,
    miMeshID: u32,
};

@group(1) @binding(1)
var<storage, read> aBVHNodes: array<BVHNode2>;

@group(1) @binding(2)
var<storage, read> aSceneVertexPositions: array<vec4<f32>>;

@group(1) @binding(3)
var<storage, read> aTrianglePositionIndices: array<u32>;

@group(1) @binding(4)
var<storage, read> aDynamicBVHNodes: array<BVHNode2>;

@group(1) @binding(5)
var<storage, read> aDynamicVertexPositions: array<vec4<f32>>;

@group(1) @binding(6)
var<storage, read> aiDynamicTriangleIndices: array<u32>;

@group(1) @binding(7)
var blueNoiseTexture: texture_2d<f32>;

@group(1) @binding(8)
var<uniform> defaultUniformData: DefaultUniformData;

@vertex
fn vs_main(in: VertexInput) -> VertexOutput 
{
    var out: VertexOutput;
    out.pos = in.pos;
    out.texCoord = in.texCoord;
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> FragmentOutput 
{
    var fReservoirSize: f32 = 8.0f;

    var out: FragmentOutput;

    if(defaultUniformData.miFrame < 1)
    {
        return out;
    }

    var randomResult: RandomResult = initRand(
        u32(in.texCoord.x * 100.0f + in.texCoord.y * 200.0f) + u32(defaultUniformData.mfRand0 * 100.0f),
        u32(in.pos.x * 10.0f + in.pos.z * 20.0f) + u32(defaultUniformData.mfRand0 * 100.0f),
        10u);

    let worldPosition: vec4<f32> = textureSample(
        worldPositionTexture, 
        textureSampler, 
        in.texCoord);

    let normal: vec3<f32> = textureSample(
        normalTexture, 
        textureSampler, 
        in.texCoord).xyz;

    if(worldPosition.w <= 0.0f)
    {
        out.radianceOutput = vec4<f32>(0.0f, 0.0f, 1.0f, 0.0f);
        out.temporalReservoir = vec4<f32>(0.0f, 0.0f, 0.0f, 0.0f);
        return out;
    }

    var ambientOcclusionSample: vec4<f32>;

    let fCenterDepth: f32 = fract(worldPosition.w);
    let iCenterMeshID: i32 = i32(worldPosition.w - fCenterDepth);

    let iNumPermutations: i32 = 5;

    let origScreenCoord: vec2<i32> = vec2<i32>(
        i32(in.texCoord.x * f32(defaultUniformData.miScreenWidth)),
        i32(in.texCoord.y * f32(defaultUniformData.miScreenHeight)));

    // check for disocclusion for previous history pixel
    var fDisocclusion: f32 = 0.0f;
    let prevScreenUV: vec2<f32> = getPreviousScreenUV(in.texCoord);
    if(isPrevUVOutOfBounds(in.texCoord))
    {
        fDisocclusion = 1.0f;
    }
    else
    {
        fDisocclusion = f32(isDisoccluded2(in.texCoord, prevScreenUV));
    }
    let fValidHistory: f32 = 1.0f - fDisocclusion;

    var result: TemporalRestirResult;
    result.mReservoir = textureSample(
        prevTemporalReservoirTexture,
        textureSampler,
        prevScreenUV) * fValidHistory;
    result.mRadiance = textureSample(
        prevTemporalRadianceTexture,
        textureSampler,
        prevScreenUV) * fValidHistory;
    result.mHitPosition = textureSample(
        prevTemporalHitPositionTexture,
        textureSampler,
        prevScreenUV) * fValidHistory;
    result.mHitNormal = textureSample(
        prevTemporalHitNormalTexture,
        textureSampler,
        prevScreenUV) * fValidHistory;
    result.mRayDirection = textureSample(
        prevTemporalRayDirectionTexture,
        textureSampler,
        prevScreenUV) * fValidHistory;
    result.mIntersectionResult.miHitTriangle = UINT32_MAX;
    result.mfNumValidSamples = 0.0f;

    // more samples for disoccluded pixel
    var iNumCenterSamples: i32 = 1;
    if(fDisocclusion >= 1.0f || result.mReservoir.z <= 1.0f)
    {
        iNumCenterSamples = 4;
    }
    
    var fNumValidSamples: f32 = 0.0f;

    var screenCoord: vec2<u32> = vec2<u32>(
        u32(in.texCoord.x * f32(defaultUniformData.miScreenWidth)),
        u32(in.texCoord.y * f32(defaultUniformData.miScreenHeight))
    );
    let textureSize: vec2<u32> = textureDimensions(blueNoiseTexture, 0);

    // blue noise tile coordinate
    let iTileSize: u32 = 32u;
    let iNumTilesPerRow: u32 = textureSize.x / iTileSize;
    let iNumTotalTiles: u32 = iNumTilesPerRow * iNumTilesPerRow;
    
    let iCurrIndex: u32 = u32(defaultUniformData.miFrame) * u32(iNumCenterSamples);

    // center pixel sample
    for(var iSample: i32 = 0; iSample < iNumCenterSamples; iSample++)
    {
        var sampleRayDirection: vec3<f32> = vec3<f32>(0.0f, 0.0f, 0.0f);
        var ray: Ray;

        if((u32(defaultUniformData.miFrame) > 0u && u32(defaultUniformData.miFrame) % VALIDATION_STEP == 0u && iSample == 0 && fDisocclusion <= 0.0f)/* || fDisocclusion >= 1.0f*/)
        {
            // validation step, or just re-use the ray on disocclusion
            sampleRayDirection = result.mRayDirection.xyz;
        }
        else
        {
            let iTileX: u32 = (iCurrIndex + u32(iNumCenterSamples)) % iNumTilesPerRow;
            let iTileY: u32 = ((iCurrIndex + u32(iNumCenterSamples)) / iNumTilesPerRow) % (iNumTilesPerRow * iNumTilesPerRow);

            let iTileOffsetX: u32 = (iCurrIndex + u32(iNumCenterSamples)) % iTileSize;
            let iTileOffsetY: u32 = ((iCurrIndex + u32(iNumCenterSamples)) / iTileSize) % (iTileSize * iTileSize);

            let iOffsetX: u32 = iTileOffsetX + iTileX * iTileSize;
            let iOffsetY: u32 = iTileOffsetY + iTileY * iTileSize; 

            // sample ray for the center sample
            randomResult = nextRand(randomResult.miSeed);
            let fRand0: f32 = randomResult.mfNum;
            randomResult = nextRand(randomResult.miSeed);
            let fRand1: f32 = randomResult.mfNum;

            screenCoord.x = (screenCoord.x + iOffsetX) % textureSize.x;
            screenCoord.y = (screenCoord.y + iOffsetY) % textureSize.y;
            var sampleUV: vec2<f32> = vec2<f32>(
                f32(screenCoord.x) / f32(textureSize.x),
                f32(screenCoord.y) / f32(textureSize.y) 
            );

            var blueNoise: vec3<f32> = textureSample(
                blueNoiseTexture,
                textureSampler,
                sampleUV
            ).xyz;

            ray = uniformSampling(
                worldPosition.xyz,
                normal.xyz,
                blueNoise.x,
                blueNoise.y);
            //ray = uniformSampling(
            //    worldPosition.xyz,
            //    normal.xyz,
            //    fRand0,
            //    fRand1);
            var offsetOrigin: vec3<f32> = ray.mOrigin.xyz + normal.xyz * 0.5f;
            ray.mOrigin += vec4<f32>(offsetOrigin, 1.0f);
            sampleRayDirection = ray.mDirection.xyz;
        }

        // restir
        result.mIntersectionResult.miHitTriangle = UINT32_MAX;
        result = temporalRestir(
            result,

            worldPosition.xyz,
            normal,
            in.texCoord,
            sampleRayDirection,

            fReservoirSize,
            1.0f,
            randomResult,
            0u);
        
        ambientOcclusionSample = result.mAmbientOcclusion;

        // record intersection triangle in w component
        var fIntersection: f32 = 0.0f;
        if(result.mIntersectionResult.miHitTriangle != UINT32_MAX)
        {
            fIntersection = f32(result.mIntersectionResult.miHitTriangle);
        }
        out.rayDirection = vec4<f32>(sampleRayDirection.xyz, fIntersection);
        out.sampleRayHitPosition = vec4<f32>(result.mIntersectionResult.mHitPosition, fIntersection);
        out.sampleRayDirection = vec4<f32>(sampleRayDirection, fIntersection);
    }

    let fPlaneD: f32 = -dot(worldPosition.xyz, normal.xyz);

    // permutation samples
    for(var iSample: i32 = 1; iSample < iNumPermutations; iSample++)
    {
        var aXOR: array<vec2<i32>, 4>;
        aXOR[0] = vec2<i32>(3, 3);
        aXOR[1] = vec2<i32>(2, 1);
        aXOR[2] = vec2<i32>(1, 2);
        aXOR[3] = vec2<i32>(3, 3);
        
        var aOffsets: array<vec2<i32>, 4>;
        aOffsets[0] = vec2<i32>(-1, -1);
        aOffsets[1] = vec2<i32>(1, 1);
        aOffsets[2] = vec2<i32>(-1, 1);
        aOffsets[3] = vec2<i32>(1, -1);

        // apply permutation offset to screen coordinate, converting to uv after
        let iFrame: i32 = i32(defaultUniformData.miFrame);
        let iIndex0: i32 = iFrame & 3;
        let iIndex1: i32 = (iSample + (iFrame ^ 1)) & 3;
        let offset: vec2<i32> = aOffsets[iIndex0] + aOffsets[iIndex1];
        let screenCoord: vec2<i32> = (origScreenCoord + offset) ^ aXOR[iFrame & 3]; 
        
        var sampleRayDirection: vec3<f32> = vec3<f32>(0.0f, 0.0f, 0.0f);
        var ray: Ray;

        // get sample world position, normal, and ray direction
        var fJacobian: f32 = 1.0f;
        {
            // permutation uv
            var sampleUV: vec2<f32> = vec2<f32>(
                ceil(f32(screenCoord.x) + 0.5f) / f32(defaultUniformData.miScreenWidth),
                ceil(f32(screenCoord.y) + 0.5f) / f32(defaultUniformData.miScreenHeight));

            // back project to previous frame's screen coordinate
            var motionVector: vec2<f32> = textureSample(
                motionVectorTexture,
                textureSampler,
                sampleUV).xy;
            motionVector = motionVector * 2.0f - 1.0f;
            sampleUV -= motionVector;

            // sample world position
            let sampleWorldPosition: vec4<f32> = textureSample(
                prevWorldPositionTexture,
                textureSampler,
                sampleUV);

            let sampleNormal: vec3<f32> = textureSample(
                prevNormalTexture,
                textureSampler,
                sampleUV).xyz;

            // neighbor normal difference check 
            let fDP: f32 = dot(sampleNormal, normal);
            if(fDP <=  0.6f)
            {
                continue;
            }

            // neightbor depth difference check
            let fSampleDepth: f32 = fract(sampleWorldPosition.w);
            //let fDepthDiff: f32 = abs(fCenterDepth - fSampleDepth);
            //if(fDepthDiff >= 0.05f)
            //{
            //    continue;
            //} 

            let fPlaneDistance: f32 = dot(normal.xyz, sampleWorldPosition.xyz) + fPlaneD;
            if(abs(fPlaneDistance) >= 0.2f)
            {
                continue;
            }

            // mesh id difference check
            let iSampleMeshID: i32 = i32(floor((sampleWorldPosition.w - fSampleDepth) + 0.5f));
            if(iSampleMeshID != iCenterMeshID)
            {
                continue;
            }

            // hit point and hit normal for jacobian
            let sampleHitPoint: vec3<f32> = textureSample(
                prevTemporalHitPositionTexture,
                textureSampler,
                sampleUV).xyz;

            var neighborHitNormal: vec3<f32> = textureSample(
                prevTemporalHitNormalTexture,
                textureSampler,
                sampleUV).xyz;
            let centerToNeighborHitPointUnNormalized: vec3<f32> = sampleHitPoint - worldPosition.xyz;
            let neighborToNeighborHitPointUnNormalized: vec3<f32> = sampleHitPoint - sampleWorldPosition.xyz;
            let centerToNeighborHitPointNormalized: vec3<f32> = normalize(centerToNeighborHitPointUnNormalized);
            let neighborToNeighborHitPointNormalized: vec3<f32> = normalize(neighborToNeighborHitPointUnNormalized);
            
            // compare normals for jacobian
            let fDP0: f32 = max(dot(neighborHitNormal, centerToNeighborHitPointNormalized * -1.0f), 0.0f);
            var fDP1: f32 = max(dot(neighborHitNormal, neighborToNeighborHitPointNormalized * -1.0f), 1.0e-4f);
            fJacobian = fDP0 / fDP1;

            // compare length for jacobian 
            let fCenterToHitPointLength: f32 = length(centerToNeighborHitPointUnNormalized);
            let fNeighborToHitPointLength: f32 = length(neighborToNeighborHitPointUnNormalized);
            fJacobian *= ((fCenterToHitPointLength * fCenterToHitPointLength) / (fNeighborToHitPointLength * fNeighborToHitPointLength));
            fJacobian = clamp(fJacobian, 0.0f, 1.0f);

            sampleRayDirection = centerToNeighborHitPointNormalized;
        }

        result.mIntersectionResult.miHitTriangle = UINT32_MAX;
        result = temporalRestir(
            result,

            worldPosition.xyz,
            normal,
            in.texCoord,
            sampleRayDirection,

            fReservoirSize,
            fJacobian,
            randomResult,
            u32(iSample));

    }   // for sample = 0 to num samples   

    result.mReservoir.w = clamp(result.mReservoir.x / max(result.mReservoir.z * result.mReservoir.y, 0.001f), 0.0f, 1.0f);

    var fReservoirWeight: f32 = result.mReservoir.w;
    
    out.radianceOutput = result.mRadiance;// * fReservoirWeight;
    out.temporalReservoir = result.mReservoir;
    out.rayDirection = result.mRayDirection;

    // debug disocclusion
    var prevInputTexCoord: vec2<f32> = getPreviousScreenUV(in.texCoord);
    out.rayDirection.w = f32(isDisoccluded2(in.texCoord, prevInputTexCoord));

    let fAO: f32 = 1.0f - (ambientOcclusionSample.x / ambientOcclusionSample.y);
    
    out.ambientOcclusionOutput = vec4<f32>(ambientOcclusionSample.x, ambientOcclusionSample.y, fAO, 1.0f);
    out.hitPosition = result.mHitPosition;
    out.hitNormal = result.mHitNormal;
    
    return out;
}

/////
fn temporalRestir(
    prevResult: TemporalRestirResult,

    worldPosition: vec3<f32>,
    normal: vec3<f32>,
    inputTexCoord: vec2<f32>,
    rayDirection: vec3<f32>,

    fMaxTemporalReservoirSamples: f32,
    fJacobian: f32,
    randomResult: RandomResult,
    iSampleIndex: u32) -> TemporalRestirResult
{
    let fOneOverPDF: f32 = 1.0f / PI;

    var ret: TemporalRestirResult = prevResult;
    ret.mRandomResult = randomResult;
    
    ret.mRandomResult = nextRand(ret.mRandomResult.miSeed);
    let fRand0: f32 = ret.mRandomResult.mfNum;
    ret.mRandomResult = nextRand(ret.mRandomResult.miSeed);
    let fRand1: f32 = ret.mRandomResult.mfNum;
    ret.mRandomResult = nextRand(ret.mRandomResult.miSeed);
    let fRand2: f32 = ret.mRandomResult.mfNum;

    // reset the ambient occlusion counts on disoccluded pixels
    var prevAmbientOcclusion: vec4<f32> = vec4<f32>(0.0f, 0.0f, 0.0f, 0.0f);
    var iDisoccluded: i32 = 1;

    // get the non-disoccluded and non-out-of-bounds pixel
    var prevInputTexCoord: vec2<f32> = getPreviousScreenUV(inputTexCoord);
    if(!isPrevUVOutOfBounds(inputTexCoord) && !isDisoccluded2(inputTexCoord, prevInputTexCoord))
    {
        prevAmbientOcclusion = textureSample(
            prevAmbientOcclusionTexture,
            textureSampler,
            prevInputTexCoord
        );

        iDisoccluded = 0;
    }

    var prevTemporalReservoir: vec4<f32> = textureSample(
        prevTemporalReservoirTexture,
        textureSampler,
        prevInputTexCoord
    );

    var fAmbientOcclusionHit: f32 = prevAmbientOcclusion.x;
    let fAmbientOcclusionCount: f32 = prevAmbientOcclusion.y + 1.0f;

    var ray: Ray;
    ray.mOrigin = vec4<f32>(worldPosition + normal * 0.02f, 1.0f);
    ray.mDirection = vec4<f32>(rayDirection, 1.0f);

    //let intersectionInfo: IntersectBVHResult = intersectBVH(ray);
    //var intersectionInfo: IntersectBVHResult;
    //for(var iMeshNodeIndex: u32 = bvhProcessInfo.miStartNodeIndex; iMeshNodeIndex <= bvhProcessInfo.miEndNodeIndex; iMeshNodeIndex++)
    //{
    //    intersectionInfo = intersectBVH2(ray, iMeshNodeIndex);
    //    if(intersectionInfo.miHitTriangle != UINT32_MAX)
    //    {
    //        break;
    //    }
    //}
    //let intersectionInfo: IntersectBVHResult = intersectBVH3(ray, 0u);
    let intersectionInfo: IntersectBVHResult = intersectBVH4(ray, 0u);

    var candidateHitPosition: vec4<f32> = vec4<f32>(intersectionInfo.mHitPosition.xyz, 1.0f);
    var candidateHitNormal: vec4<f32> = vec4<f32>(intersectionInfo.mHitNormal.xyz, 1.0f);

    var candidateRadiance: vec4<f32> = vec4<f32>(0.0f, 0.0f, 0.0f, 1.0f);
    var candidateRayDirection: vec4<f32> = vec4<f32>(rayDirection, 1.0f);
    var fRadianceDP: f32 = max(dot(normal, ray.mDirection.xyz), 0.0f);
    var fDistanceAttenuation: f32 = 1.0f;
    if(intersectionInfo.miHitTriangle == UINT32_MAX)
    {
        // didn't hit anything, use skylight
        let skyUV: vec2<f32> = octahedronMap2(ray.mDirection.xyz);
        
        candidateRadiance = textureSample(
            skyTexture,
            textureSampler,
            skyUV);

        candidateHitNormal = ray.mDirection * -1.0f;
        candidateHitPosition.x = worldPosition.x + ray.mDirection.x * 50000.0f;
        candidateHitPosition.y = worldPosition.y + ray.mDirection.y * 50000.0f;
        candidateHitPosition.z = worldPosition.z + ray.mDirection.z * 50000.0f;
    }
    else
    {
        var hitPosition: vec3<f32> = intersectionInfo.mHitPosition.xyz;
        
        // distance for on-screen radiance and ambient occlusion
        var fDistance: f32 = length(hitPosition - worldPosition);
        if(fDistance < defaultUniformData.mfAmbientOcclusionDistanceThreshold)
        {
            fAmbientOcclusionHit += 1.0f;
        }

        fDistanceAttenuation = 1.0f / max(fDistance * fDistance, 1.0f);
        if(fDistance < 1.0f)
        {
            fDistanceAttenuation = fDistance;
        }

        // get on-screen radiance if there's any
        var clipSpacePosition: vec4<f32> = vec4<f32>(hitPosition, 1.0) * defaultUniformData.mViewProjectionMatrix;
        clipSpacePosition.x /= clipSpacePosition.w;
        clipSpacePosition.y /= clipSpacePosition.w;
        clipSpacePosition.z /= clipSpacePosition.w;
        clipSpacePosition.x = clipSpacePosition.x * 0.5f + 0.5f;
        clipSpacePosition.y = 1.0f - (clipSpacePosition.y * 0.5f + 0.5f);
        clipSpacePosition.z = clipSpacePosition.z * 0.5f + 0.5f;

        let worldSpaceHitPosition: vec4<f32> = textureSample(
            worldPositionTexture,
            textureSampler,
            clipSpacePosition.xy);
        var hitPositionClipSpace: vec4<f32> = vec4<f32>(worldSpaceHitPosition.xyz, 1.0f) * defaultUniformData.mViewProjectionMatrix;
        hitPositionClipSpace.x /= hitPositionClipSpace.w;
        hitPositionClipSpace.y /= hitPositionClipSpace.w;
        hitPositionClipSpace.z /= hitPositionClipSpace.w;
        hitPositionClipSpace.x = hitPositionClipSpace.x * 0.5f + 0.5f;
        hitPositionClipSpace.y = 1.0f - (hitPositionClipSpace.y * 0.5f + 0.5f);
        hitPositionClipSpace.z = hitPositionClipSpace.z * 0.5f + 0.5f;

        let fDepthDiff: f32 = hitPositionClipSpace.z - clipSpacePosition.z;

        if(clipSpacePosition.x >= 0.0f && clipSpacePosition.x <= 1.0f &&
           clipSpacePosition.y >= 0.0f && clipSpacePosition.y <= 1.0f && 
           fDepthDiff >= 0.1f)
        {
            // on screen

            // back project to previous frame's clip space coordinate
            var motionVector: vec2<f32> = textureSample(
                motionVectorTexture,
                textureSampler,
                inputTexCoord).xy;
            motionVector = motionVector * 2.0f - 1.0f;
            clipSpacePosition.x -= motionVector.x;
            clipSpacePosition.y -= motionVector.y;

            // distance attenuation
            let fReflectivity: f32 = 0.2f;

            candidateRadiance = textureSample(
                prevTemporalRadianceTexture,
                textureSampler,
                clipSpacePosition.xy);

            candidateRadiance.x = (candidateRadiance.x * fReflectivity);
            candidateRadiance.y = (candidateRadiance.y * fReflectivity);
            candidateRadiance.z = (candidateRadiance.z * fReflectivity);
        }
        else 
        {
            let fReflectivity: f32 = 0.2f;

            // not on screen, check irradiance cache
            let iHitIrradianceCacheIndex: u32 = fetchIrradianceCacheIndex(hitPosition);
            let irradianceCachePosition: vec4<f32> = getIrradianceCachePosition(iHitIrradianceCacheIndex);
            if(irradianceCachePosition.w > 0.0f)
            {
                let positionToCacheDirection: vec3<f32> = normalize(irradianceCachePosition.xyz - worldPosition) * -1.0f; 
                let irradianceCacheProbeRadiance: vec3<f32> = getRadianceFromIrradianceCacheProbe(
                    positionToCacheDirection, 
                    iHitIrradianceCacheIndex);

                candidateRadiance.x = irradianceCacheProbeRadiance.x * fReflectivity;
                candidateRadiance.y = irradianceCacheProbeRadiance.y * fReflectivity;
                candidateRadiance.z = irradianceCacheProbeRadiance.z * fReflectivity;
            }
        }
    }    

    candidateRadiance.x = candidateRadiance.x * fJacobian * fRadianceDP * fDistanceAttenuation * fOneOverPDF;
    candidateRadiance.y = candidateRadiance.y * fJacobian * fRadianceDP * fDistanceAttenuation * fOneOverPDF;
    candidateRadiance.z = candidateRadiance.z * fJacobian * fRadianceDP * fDistanceAttenuation * fOneOverPDF;

    // reservoir
    let fLuminance: f32 = computeLuminance(
        ToneMapFilmic_Hejl2015(
            candidateRadiance.xyz, 
            max(max(defaultUniformData.mLightRadiance.x, defaultUniformData.mLightRadiance.y), defaultUniformData.mLightRadiance.z)
        )
    );

    var fM: f32 = 1.0f;
    var fPHat: f32 = clamp(fLuminance, 0.0f, 1.0f);
    //if(iSampleIndex > 0u)
    //{
    //    fM = 0.25f;
    //    fPHat = clamp(fLuminance * prevTemporalReservoir.z * 0.25f * prevTemporalReservoir.w, 0.0f, 1.0f);
    //}
    var updateResult: ReservoirResult = updateReservoir(
        prevResult.mReservoir,
        fPHat,
        fM,
        fRand2);
    
    if(updateResult.mbExchanged)
    {
        ret.mRadiance = candidateRadiance;
        ret.mHitPosition = candidateHitPosition;
        ret.mHitNormal = candidateHitNormal;
        ret.mRayDirection = candidateRayDirection;
    }

    // clamp reservoir
    if(updateResult.mReservoir.z > fMaxTemporalReservoirSamples)
    {
        let fPct: f32 = fMaxTemporalReservoirSamples / updateResult.mReservoir.z;
        updateResult.mReservoir.x *= fPct;
        updateResult.mReservoir.z = fMaxTemporalReservoirSamples;
    }
    
    ret.mReservoir = updateResult.mReservoir;
    ret.mAmbientOcclusion = vec4<f32>(fAmbientOcclusionHit, fAmbientOcclusionCount, f32(iDisoccluded), 0.0f);
    ret.mIntersectionResult = intersectionInfo;

    ret.mfNumValidSamples += fM * f32(fLuminance > 0.0f);
    
    return ret;
}

/////
fn updateReservoir(
    reservoir: vec4<f32>,
    fPHat: f32,
    fM: f32,
    fRand: f32) -> ReservoirResult
{
    var ret: ReservoirResult;
    ret.mReservoir = reservoir;
    ret.mbExchanged = false;

    ret.mReservoir.x += fPHat;

    var fMult: f32 = clamp(fPHat, 0.3f, 1.0f); 
    ret.mReservoir.z += fM * fMult;
    
    var fWeightPct: f32 = fPHat / ret.mReservoir.x;

    if(fRand < fWeightPct || reservoir.z <= 0.0f)
    {
        ret.mReservoir.y = fPHat;
        ret.mbExchanged = true;
    }

    return ret;
}

/////
fn initRand(
    val0: u32, 
    val1: u32, 
    backoff: u32) -> RandomResult
{
    var retResult: RandomResult;

    var v0: u32 = val0;
    var v1: u32 = val1;
    var s0: u32 = 0u;

    for(var n: u32 = 0; n < backoff; n++)
    {
        s0 += u32(0x9e3779b9);
        v0 += ((v1 << 4) + 0xa341316c) ^ (v1 + s0) ^ ((v1 >> 5) + 0xc8013ea4);
        v1 += ((v0 << 4) + 0xad90777d) ^ (v0 + s0) ^ ((v0 >> 5) + 0x7e95761e);
    }

    retResult.miSeed = v0;
    retResult.mfNum = 0.0f;

    return retResult;
}

/////
fn nextRand(s: u32) -> RandomResult
{
    var retResult: RandomResult;

    var sCopy: u32 = s;
    sCopy = (1664525u * sCopy + 1013904223u);
    retResult.mfNum = f32(sCopy & 0x00FFFFFF) / f32(0x01000000);
    retResult.miSeed = sCopy;

    return retResult;
}

/////
fn uniformSampling(
    worldPosition: vec3<f32>,
    normal: vec3<f32>,
    fRand0: f32,
    fRand1: f32) -> Ray
{
    let fPhi: f32 = 2.0f * PI * fRand0;
    let fCosTheta: f32 = 1.0f - fRand1;
    let fSinTheta: f32 = sqrt(1.0f - fCosTheta * fCosTheta);
    let h: vec3<f32> = vec3<f32>(
        cos(fPhi) * fSinTheta,
        sin(fPhi) * fSinTheta,
        fCosTheta);

    var up: vec3<f32> = vec3<f32>(0.0f, 1.0f, 0.0f);
    if(abs(normal.y) > 0.999f)
    {
        up = vec3<f32>(1.0f, 0.0f, 0.0f);
    }
    let tangent: vec3<f32> = normalize(cross(up, normal));
    let binormal: vec3<f32> = normalize(cross(normal, tangent));
    let rayDirection: vec3<f32> = normalize(tangent * h.x + binormal * h.y + normal * h.z);

    var ray: Ray;
    ray.mOrigin = vec4<f32>(worldPosition, 1.0f);
    ray.mDirection = vec4<f32>(rayDirection, 1.0f);
    ray.mfT = vec4<f32>(FLT_MAX, FLT_MAX, FLT_MAX, FLT_MAX);

    return ray;
}

/////
fn computeLuminance(
    radiance: vec3<f32>) -> f32
{
    return dot(radiance, vec3<f32>(0.2126f, 0.7152f, 0.0722f));
}

/////
fn octahedronMap2(direction: vec3<f32>) -> vec2<f32>
{
    let fDP: f32 = dot(vec3<f32>(1.0f, 1.0f, 1.0f), abs(direction));
    let newDirection: vec3<f32> = vec3<f32>(direction.x, direction.z, direction.y) / fDP;

    var ret: vec2<f32> =
        vec2<f32>(
            (1.0f - abs(newDirection.z)) * sign(newDirection.x),
            (1.0f - abs(newDirection.x)) * sign(newDirection.z));
       
    if(newDirection.y < 0.0f)
    {
        ret = vec2<f32>(
            newDirection.x, 
            newDirection.z);
    }

    ret = ret * 0.5f + vec2<f32>(0.5f, 0.5f);
    ret.y = 1.0f - ret.y;

    return ret;
}

/////
fn barycentric(
    p: vec3<f32>, 
    a: vec3<f32>, 
    b: vec3<f32>, 
    c: vec3<f32>) -> vec3<f32>
{
    let v0: vec3<f32> = b - a;
    let v1: vec3<f32> = c - a;
    let v2: vec3<f32> = p - a;
    let fD00: f32 = dot(v0, v0);
    let fD01: f32 = dot(v0, v1);
    let fD11: f32 = dot(v1, v1);
    let fD20: f32 = dot(v2, v0);
    let fD21: f32 = dot(v2, v1);
    let fOneOverDenom: f32 = 1.0f / (fD00 * fD11 - fD01 * fD01);
    let fV: f32 = (fD11 * fD20 - fD01 * fD21) * fOneOverDenom;
    let fW: f32 = (fD00 * fD21 - fD01 * fD20) * fOneOverDenom;
    let fU: f32 = 1.0f - fV - fW;

    return vec3<f32>(fU, fV, fW);
}


/////
fn rayPlaneIntersection(
    pt0: vec3<f32>,
    pt1: vec3<f32>,
    planeNormal: vec3<f32>,
    fPlaneDistance: f32) -> f32
{
    var fRet: f32 = FLT_MAX;
    let v: vec3<f32> = pt1 - pt0;

    let fDenom: f32 = dot(v, planeNormal);
    fRet = -(dot(pt0, planeNormal) + fPlaneDistance) / (fDenom + 1.0e-5f);

    return fRet;
}

/////
fn rayBoxIntersect(
    rayPosition: vec3<f32>,
    rayDir: vec3<f32>,
    bboxMin: vec3<f32>,
    bboxMax: vec3<f32>) -> bool
{
    //let oneOverRay: vec3<f32> = 1.0f / rayDir.xyz;
    let tMin: vec3<f32> = (bboxMin - rayPosition) / rayDir.xyz;
    let tMax: vec3<f32> = (bboxMax - rayPosition) / rayDir.xyz;

    var fTMin: f32 = min(tMin.x, tMax.x);
    var fTMax: f32 = max(tMin.x, tMax.x);

    fTMin = max(fTMin, min(tMin.y, tMax.y));
    fTMax = min(fTMax, max(tMin.y, tMax.y));

    fTMin = max(fTMin, min(tMin.z, tMax.z));
    fTMax = min(fTMax, max(tMin.z, tMax.z));

    return fTMax >= fTMin;
}



/////
fn rayTriangleIntersection(
    rayPt0: vec3<f32>, 
    rayPt1: vec3<f32>, 
    triPt0: vec3<f32>, 
    triPt1: vec3<f32>, 
    triPt2: vec3<f32>) -> RayTriangleIntersectionResult
{
    var ret: RayTriangleIntersectionResult;

    let v0: vec3<f32> = normalize(triPt1 - triPt0);
    let v1: vec3<f32> = normalize(triPt2 - triPt0);
    let cp: vec3<f32> = cross(v0, v1);

    let triNormal: vec3<f32> = normalize(cp);
    let fPlaneDistance: f32 = -dot(triPt0, triNormal);

    let fT: f32 = rayPlaneIntersection(
        rayPt0, 
        rayPt1, 
        triNormal, 
        fPlaneDistance);
    if(fT < 0.0f)
    {
        ret.mIntersectPosition = vec3<f32>(FLT_MAX, FLT_MAX, FLT_MAX);
        return ret;
    }

    let collisionPt: vec3<f32> = rayPt0 + (rayPt1 - rayPt0) * fT;
    
    let edge0: vec3<f32> = normalize(triPt1 - triPt0);
    let edge1: vec3<f32> = normalize(triPt2 - triPt0);
    let edge2: vec3<f32> = normalize(triPt0 - triPt2);

    // edge 0
    var C: vec3<f32> = cross(edge0, normalize(collisionPt - triPt0));
    if(dot(triNormal, C) < 0.0f)
    {
        ret.mIntersectPosition = vec3<f32>(FLT_MAX, FLT_MAX, FLT_MAX);
        return ret;
    }

    // edge 1
    C = cross(edge1, normalize(collisionPt - triPt1));
    if(dot(triNormal, C) < 0.0f)
    {
        ret.mIntersectPosition = vec3<f32>(FLT_MAX, FLT_MAX, FLT_MAX);
        return ret;
    }

    // edge 2
    C = cross(edge2, normalize(collisionPt - triPt2));
    if(dot(triNormal, C) < 0.0f)
    {
        ret.mIntersectPosition = vec3<f32>(FLT_MAX, FLT_MAX, FLT_MAX);
        return ret;
    }
    
    ret.mBarycentricCoordinate = barycentric(collisionPt, triPt0, triPt1, triPt2);
    ret.mIntersectPosition = (triPt0 * ret.mBarycentricCoordinate.x + triPt1 * ret.mBarycentricCoordinate.y + triPt2 * ret.mBarycentricCoordinate.z);
    ret.mIntersectNormal = triNormal.xyz;

    return ret;
}

/////
fn isDisoccluded(
    inputTexCoord: vec2<f32>) -> bool
{
    // world position, normal, and motion vector
    let worldPosition = textureSample(
        worldPositionTexture,
        textureSampler,
        inputTexCoord);
    let normal = textureSample(
        normalTexture,
        textureSampler,
        inputTexCoord);
    var motionVector: vec4<f32> = textureSample(
        motionVectorTexture,
        textureSampler,
        inputTexCoord);
    motionVector.x = motionVector.x * 2.0f - 1.0f;
    motionVector.y = motionVector.y * 2.0f - 1.0f;
    
    let iMesh: u32 = u32(ceil(motionVector.z - 0.5f)) - 1;

    // world position, normal, motion vector from previous frame with back projected uv
    var backProjectedScreenUV: vec2<f32> = inputTexCoord - motionVector.xy;
    if(backProjectedScreenUV.x < 0.0f || backProjectedScreenUV.y < 0.0f || 
       backProjectedScreenUV.x > 1.0f || backProjectedScreenUV.y > 1.0f)
    {
        return true;
    }

    var prevWorldPosition: vec4<f32> = textureSample(
        prevWorldPositionTexture,
        textureSampler,
        backProjectedScreenUV
    );
    var prevNormal: vec4<f32> = textureSample(
        prevNormalTexture,
        textureSampler,
        backProjectedScreenUV
    );
    var prevMotionVectorAndMeshIDAndDepth: vec4<f32> = textureSample(
        prevMotionVectorTexture,
        textureSampler,
        backProjectedScreenUV
    );

    var fOneOverScreenHeight: f32 = 1.0f / f32(defaultUniformData.miScreenHeight);
    var fOneOverScreenWidth: f32 = 1.0f / f32(defaultUniformData.miScreenWidth);

    var bestBackProjectedScreenUV: vec2<f32> = backProjectedScreenUV;
    var fShortestWorldDistance: f32 = FLT_MAX;
    for(var iY: i32 = -1; iY <= 1; iY++)
    {
        var fSampleY: f32 = backProjectedScreenUV.y + f32(iY) * fOneOverScreenHeight;
        fSampleY = clamp(fSampleY, 0.0f, 1.0f);
        for(var iX: i32 = -1; iX <= 1; iX++)
        {
            var fSampleX: f32 = backProjectedScreenUV.x + f32(iX) * fOneOverScreenWidth;
            fSampleX = clamp(fSampleX, 0.0f, 1.0f);

            let sampleUV: vec2<f32> = vec2<f32>(fSampleX, fSampleY);
            var checkPrevWorldPosition: vec4<f32> = textureSample(
                prevWorldPositionTexture,
                textureSampler,
                sampleUV
            );

            var checkPrevNormal: vec4<f32> = textureSample(
                prevNormalTexture,
                textureSampler,
                sampleUV
            );

            var worldPositionDiff: vec3<f32> = checkPrevWorldPosition.xyz - worldPosition.xyz;
            var fCheckWorldPositionDistance: f32 = dot(worldPositionDiff, worldPositionDiff);
            var fCheckNormalDP: f32 = abs(dot(checkPrevNormal.xyz, normal.xyz));
            if(fCheckWorldPositionDistance < fShortestWorldDistance && fCheckNormalDP >= 0.98f)
            {
                fShortestWorldDistance = fCheckWorldPositionDistance;
                bestBackProjectedScreenUV = sampleUV;
            }
        }
    }

    backProjectedScreenUV = bestBackProjectedScreenUV;
    prevWorldPosition = textureSample(
        prevWorldPositionTexture,
        textureSampler,
        backProjectedScreenUV
    );
    prevNormal = textureSample(
        prevNormalTexture,
        textureSampler,
        backProjectedScreenUV
    );
    prevMotionVectorAndMeshIDAndDepth = textureSample(
        prevMotionVectorTexture,
        textureSampler,
        backProjectedScreenUV
    );
    
    let toCurrentDir: vec3<f32> = worldPosition.xyz - prevWorldPosition.xyz;
    //let fPlaneDistance: f32 = abs(dot(toCurrentDir, normal.xyz)); 
    let fPrevPlaneDistance: f32 = abs(dot(prevWorldPosition.xyz, normal.xyz)) - abs(dot(worldPosition.xyz, normal.xyz));

    // compute difference in world position, depth, and angle from previous frame
    
    var fDepth: f32 = motionVector.w;
    var fPrevDepth: f32 = prevMotionVectorAndMeshIDAndDepth.w;
    var fCheckDepth: f32 = abs(fDepth - fPrevDepth);
    var worldPositionDiff: vec3<f32> = prevWorldPosition.xyz - worldPosition.xyz;
    var fCheckDP: f32 = abs(dot(normalize(normal.xyz), normalize(prevNormal.xyz)));
    let iPrevMesh: u32 = u32(ceil(prevMotionVectorAndMeshIDAndDepth.z - 0.5f)) - 1;
    var fCheckWorldPositionDistance: f32 = dot(worldPositionDiff, worldPositionDiff);

    return !(iMesh == iPrevMesh && fCheckDepth <= 0.004f && fCheckWorldPositionDistance <= 0.001f && fCheckDP >= 0.99f);
    //return !(iMesh == iPrevMesh && fCheckWorldPositionDistance <= 0.00025f && fCheckDP >= 0.99f);
    //return !(iMesh == iPrevMesh && fPrevPlaneDistance <= 0.008f && fCheckDP >= 0.99f);
}

/////
fn murmurHash13(
    src: vec3<u32>) -> u32
{
    var srcCopy: vec3<u32> = src;
    var M: u32 = 0x5bd1e995u;
    var h: u32 = 1190494759u;
    srcCopy *= M; srcCopy.x ^= srcCopy.x >> 24u; srcCopy.y ^= srcCopy.y >> 24u; srcCopy.z ^= srcCopy.z >> 24u; srcCopy *= M;
    h *= M; h ^= srcCopy.x; h *= M; h ^= srcCopy.y; h *= M; h ^= srcCopy.z;
    h ^= h >> 13u; h *= M; h ^= h >> 15u;
    return h;
}

/////
fn hash13(
    src: vec3<f32>,
    iNumSlots: u32) -> u32
{
    let srcU32: vec3<u32> = vec3<u32>(
        bitcast<u32>(src.x),
        bitcast<u32>(src.y),
        bitcast<u32>(src.z)
    );

    let h: u32 = u32(murmurHash13(srcU32));
    var fValue: f32 = bitcast<f32>((h & 0x007ffffffu) | 0x3f800000u) - 1.0f;
    let iRet: u32 = clamp(u32(fValue * f32(iNumSlots - 1)), 0u, iNumSlots - 1);
    return iRet;
}

/////
fn fetchIrradianceCacheIndex(
    position: vec3<f32>
) -> u32
{
    var scaledPosition: vec3<f32> = position * 10.0f;
    let iHashKey: u32 = hash13(
        scaledPosition,
        5000u
    );

    return iHashKey;
}

/////
fn getRadianceFromIrradianceCacheProbe(
    rayDirection: vec3<f32>,
    iIrradianceCacheIndex: u32
) -> vec3<f32>
{
    if(irradianceCache[iIrradianceCacheIndex].mPosition.w == 0.0f)
    {
        return vec3<f32>(0.0f, 0.0f, 0.0f);
    }

    let probeImageUV: vec2<f32> = octahedronMap2(rayDirection);
    var iImageY: u32 = clamp(u32(probeImageUV.y * f32(PROBE_IMAGE_SIZE)), 0u, PROBE_IMAGE_SIZE - 1u);
    var iImageX: u32 = clamp(u32(probeImageUV.x * f32(PROBE_IMAGE_SIZE)), 0u, PROBE_IMAGE_SIZE - 1u);
    var iImageIndex: u32 = iImageY * PROBE_IMAGE_SIZE + iImageX;

    return irradianceCache[iIrradianceCacheIndex].mImageProbe[iImageIndex].xyz;
}

/////
fn getIrradianceCachePosition(
    iIrradianceCacheIndex: u32
) -> vec4<f32>
{
    return irradianceCache[iIrradianceCacheIndex].mPosition;
}



// -------------------------------------
// Defines
const EPS: f32 =               1e-6f;
const PLANET_RADIUS: f32 =       6371000.0f;
const PLANET_CENTER: vec3<f32> =       vec3<f32>(0.0f, -PLANET_RADIUS, 0.0f);
const ATMOSPHERE_HEIGHT: f32 =   100000.0f;
const RAYLEIGH_HEIGHT: f32 =     (ATMOSPHERE_HEIGHT * 0.08f);
const MIE_HEIGHT: f32 =          (ATMOSPHERE_HEIGHT * 0.012f);

// -------------------------------------
// Coefficients
const C_RAYLEIGH: vec3<f32> =          (vec3<f32>(5.802f, 13.558f, 33.100f) * 1e-6f);
const C_MIE: vec3<f32> =               (vec3<f32>(3.996f,  3.996f,  3.996f) * 1e-6f);
const C_OZONE: vec3<f32> =             (vec3<f32>(0.650f,  1.881f,  0.085f) * 1e-6f);

const ATMOSPHERE_DENSITY: f32 =  1.0f;
const EXPOSURE: f32 =            20.0f;

// -------------------------------------
// Math
fn SphereIntersection(
	rayStart: vec3<f32>, 
	rayDir: vec3<f32>, 
	sphereCenter: vec3<f32>, 
	sphereRadius: f32) -> vec2<f32>
{
	let rayStartDiff: vec3<f32> = rayStart - sphereCenter;
	let a: f32 = dot(rayDir, rayDir);
	let b: f32 = 2.0f * dot(rayStartDiff, rayDir);
	let c: f32 = dot(rayStartDiff, rayStartDiff) - (sphereRadius * sphereRadius);
	var d: f32 = b * b - 4.0f * a * c;
	if(d < 0)
	{
		return vec2<f32>(-1.0f, -1.0f);
	}
	else
	{
		d = sqrt(d);
		return vec2<f32>(-b - d, -b + d) / (2.0f * a);
	}
}
fn PlanetIntersection(
	rayStart: vec3<f32>, 
	rayDir: vec3<f32>) -> vec2<f32>
{
	return SphereIntersection(rayStart, rayDir, PLANET_CENTER, PLANET_RADIUS);
}
fn AtmosphereIntersection(
	rayStart: vec3<f32>, 
	rayDir: vec3<f32>) -> vec2<f32>
{
	return SphereIntersection(rayStart, rayDir, PLANET_CENTER, PLANET_RADIUS + ATMOSPHERE_HEIGHT);
}

// -------------------------------------
// Phase functions
fn PhaseRayleigh(costh: f32) -> f32
{
	return 3.0f * (1.0f + costh * costh) / (16.0f * PI);
}
fn PhaseMie(
    costh: f32, 
    g: f32) -> f32
{
	let gCopy: f32 = min(g, 0.9381f);
	let k: f32 = 1.55f * gCopy - 0.55f * gCopy * gCopy * gCopy;
	let kcosth: f32 = k * costh;
	return (1.0f - k * k) / ((4.0f * PI) * (1.0f - kcosth) * (1.0f - kcosth));
}

// -------------------------------------
// Atmosphere
fn AtmosphereHeight(positionWS: vec3<f32>) -> f32
{
	//return distance(positionWS, PLANET_CENTER) - PLANET_RADIUS;

	return length(positionWS - PLANET_CENTER) - PLANET_RADIUS;
}
fn DensityRayleigh(h: f32) -> f32
{
	return exp(-max(0.0f, h / RAYLEIGH_HEIGHT));
}
fn DensityMie(h: f32) -> f32
{
	return exp(-max(0.0f, h / MIE_HEIGHT));
}
fn DensityOzone(h: f32) -> f32
{
	// The ozone layer is represented as a tent function with a width of 30km, centered around an altitude of 25km.
	return max(0.0f, 1.0f - abs(h - 25000.0f) / 15000.0f);
}
fn AtmosphereDensity(h: f32) -> vec3<f32>
{
	return vec3<f32>(DensityRayleigh(h), DensityMie(h), DensityOzone(h));
}

// Optical depth is a unitless measurement of the amount of absorption of a participating medium (such as the atmosphere).
// This function calculates just that for our three atmospheric elements:
// R: Rayleigh
// G: Mie
// B: Ozone
// If you find the term "optical depth" confusing, you can think of it as "how much density was found along the ray in total".
fn IntegrateOpticalDepth(
    rayStart: vec3<f32>, 
    rayDir: vec3<f32>) -> vec3<f32>
{
	let intersection: vec2<f32> = AtmosphereIntersection(rayStart, rayDir);
	let  rayLength: f32 = intersection.y;

	let    sampleCount: i32 = 8;
	let  stepSize: f32 = rayLength / f32(sampleCount);

	var opticalDepth: vec3<f32> = vec3<f32>(0.0f, 0.0f, 0.0f);

	for(var i: i32 = 0; i < sampleCount; i++)
	{
		let localPosition: vec3<f32> = rayStart + rayDir * (f32(i) + 0.5f) * stepSize;
		let  localHeight: f32 = AtmosphereHeight(localPosition);
		let localDensity: vec3<f32> = AtmosphereDensity(localHeight);

		opticalDepth += localDensity * stepSize;
	}

	return opticalDepth;
}

// Calculate a luminance transmittance value from optical depth.
fn Absorb(opticalDepth: vec3<f32>) -> vec3<f32>
{
	// Note that Mie results in slightly more light absorption than scattering, about 10%
	let ret: vec3<f32> = ((C_RAYLEIGH * opticalDepth.x) + (C_MIE * (opticalDepth.y * 1.1f)) + (C_OZONE * opticalDepth.z)) * ATMOSPHERE_DENSITY * -1.0f;
	return vec3<f32>(
		exp(ret.x),
		exp(ret.y),
		exp(ret.z));

}

struct ScatteringResult
{
    radiance: vec3<f32>,
    transmittance: vec3<f32>,
};

// Integrate scattering over a ray for a single directional light source.
// Also return the transmittance for the same ray as we are already calculating the optical depth anyway.
fn IntegrateScattering(
	rayStart: vec3<f32>, 
	rayDir: vec3<f32>, 
	rayLength: f32, 
	lightDir: vec3<f32>, 
	lightColor: vec3<f32>, 
	transmittance: vec3<f32>) -> ScatteringResult
{
	// We can reduce the number of atmospheric samples required to converge by spacing them exponentially closer to the camera.
	// This breaks space view however, so let's compensate for that with an exponent that "fades" to 1 as we leave the atmosphere.
	var rayStartCopy: vec3<f32> = rayStart;
    let  rayHeight: f32 = AtmosphereHeight(rayStartCopy);
	let  sampleDistributionExponent: f32 = 1.0f + clamp(1.0f - rayHeight / ATMOSPHERE_HEIGHT, 0.0f, 1.0f) * 8; // Slightly arbitrary max exponent of 9

	let intersection: vec2<f32> = AtmosphereIntersection(rayStartCopy, rayDir);
	var rayLengthCopy: f32 = min(rayLength, intersection.y);
	if(intersection.x > 0)
	{
		// Advance ray to the atmosphere entry point
		rayStartCopy += rayDir * intersection.x;
		rayLengthCopy -= intersection.x;
	}

	let  costh: f32 = dot(rayDir, lightDir);
	let  phaseR: f32 = PhaseRayleigh(costh);
	let  phaseM: f32 = PhaseMie(costh, 0.85f);

	let    sampleCount: i32 = 64;

	var opticalDepth: vec3<f32> = vec3<f32>(0.0f, 0.0f, 0.0f);
	var rayleigh: vec3<f32> = vec3<f32>(0.0f, 0.0f, 0.0f);
	var mie: vec3<f32> = vec3<f32>(0.0f, 0.0f, 0.0f);

	var  prevRayTime: f32 = 0.0f;

	for(var i: i32 = 0; i < sampleCount; i++)
	{
		let  rayTime: f32 = pow(f32(i) / f32(sampleCount), sampleDistributionExponent) * rayLengthCopy;
		// Because we are distributing the samples exponentially, we have to calculate the step size per sample.
		let  stepSize: f32 = (rayTime - prevRayTime);

		let localPosition: vec3<f32> = rayStartCopy + rayDir * rayTime;
		let  localHeight: f32 = AtmosphereHeight(localPosition);
		let localDensity: vec3<f32> = AtmosphereDensity(localHeight);

		opticalDepth += localDensity * stepSize;

		// The atmospheric transmittance from rayStart to localPosition
		let viewTransmittance: vec3<f32> = Absorb(opticalDepth);

		let opticalDepthlight: vec3<f32> = IntegrateOpticalDepth(localPosition, lightDir);
		// The atmospheric transmittance of light reaching localPosition
		let lightTransmittance: vec3<f32> = Absorb(opticalDepthlight);

		rayleigh += viewTransmittance * lightTransmittance * phaseR * localDensity.x * stepSize;
		mie += viewTransmittance * lightTransmittance * phaseM * localDensity.y * stepSize;

		prevRayTime = rayTime;
	}

    var ret: ScatteringResult;
	ret.transmittance = Absorb(opticalDepth);
    ret.radiance = (rayleigh * C_RAYLEIGH + mie * C_MIE) * lightColor * EXPOSURE;

	return ret;
}

struct Triangle
{
    miV0 : u32,
    miV1 : u32,
    miV2 : u32,
    mPadding : u32,

    mCentroid : vec4<f32>
};

struct LeafNode
{
    miTriangleIndex: u32,
};

struct IntermediateNode
{
    mCentroid: vec4<f32>,
    mMinBounds: vec4<f32>,
    mMaxBounds: vec4<f32>,

    miLeftNodeIndex: u32,
    miRightNodeIndex: u32,
    miLeafNode: u32,
    miMortonCode: u32,
};

struct BVHProcessInfo
{
    miStep: u32,
    miStartNodeIndex: u32,
    miEndNodeIndex: u32,
    miNumMeshes: u32, 
};

/////
fn intersectTri3(
    ray: Ray,
    iTriangleIndex: u32) -> RayTriangleIntersectionResult
{
    let iV0: u32 = aTrianglePositionIndices[iTriangleIndex * 3];
    let iV1: u32 = aTrianglePositionIndices[iTriangleIndex * 3 + 1];
    let iV2: u32 = aTrianglePositionIndices[iTriangleIndex * 3 + 2];

    let pos0: vec4<f32> = aSceneVertexPositions[iV0];
    let pos1: vec4<f32> = aSceneVertexPositions[iV1];
    let pos2: vec4<f32> = aSceneVertexPositions[iV2];

    var iIntersected: u32 = 0;
    var fT: f32 = FLT_MAX;
    let intersectionInfo: RayTriangleIntersectionResult = rayTriangleIntersection(
        ray.mOrigin.xyz,
        ray.mOrigin.xyz + ray.mDirection.xyz * 1000.0f,
        pos0.xyz,
        pos1.xyz,
        pos2.xyz);
    
    return intersectionInfo;
}

/////
fn intersectBVH3(
    ray: Ray,
    iRootNodeIndex: u32) -> IntersectBVHResult
{
    var ret: IntersectBVHResult;

    var iStackTop: i32 = 0;
    var aiStack: array<u32, 256>;
    aiStack[iStackTop] = iRootNodeIndex;

    ret.mHitPosition = vec3<f32>(FLT_MAX, FLT_MAX, FLT_MAX);
    ret.miHitTriangle = UINT32_MAX;
    var fClosestDistance: f32 = FLT_MAX;

    for(var iStep: u32 = 0u; iStep < 10000u; iStep++)
    {
        if(iStackTop < 0)
        {
            break;
        }

        let iNodeIndex: u32 = aiStack[iStackTop];
        iStackTop -= 1;

        //let node: BVHNode2 = aBVHNodes[iNodeIndex];
        if(aBVHNodes[iNodeIndex].miPrimitiveID != UINT32_MAX)
        {
            let intersectionInfo: RayTriangleIntersectionResult = intersectTri3(
                ray,
                aBVHNodes[iNodeIndex].miPrimitiveID);

            if(abs(intersectionInfo.mIntersectPosition.x) < 10000.0f)
            {
                let fDistanceToEye: f32 = length(intersectionInfo.mIntersectPosition.xyz - ray.mOrigin.xyz);
                if(fDistanceToEye < fClosestDistance)
                {
                    
                    fClosestDistance = fDistanceToEye;
                    ret.mHitPosition = intersectionInfo.mIntersectPosition.xyz;
                    ret.mHitNormal = intersectionInfo.mIntersectNormal.xyz;
                    ret.miHitTriangle = aBVHNodes[iNodeIndex].miPrimitiveID;
                    ret.mBarycentricCoordinate = intersectionInfo.mBarycentricCoordinate;
                }
            }
        }
        else
        {
            let bIntersect: bool = rayBoxIntersect(
                ray.mOrigin.xyz,
                ray.mDirection.xyz,
                aBVHNodes[iNodeIndex].mMinBound.xyz,
                aBVHNodes[iNodeIndex].mMaxBound.xyz);

            // node left and right child to stack
            if(bIntersect)
            {
                iStackTop += 1;
                aiStack[iStackTop] = aBVHNodes[iNodeIndex].miChildren0;
                iStackTop += 1;
                aiStack[iStackTop] = aBVHNodes[iNodeIndex].miChildren1;
            }
        }
    }

    return ret;
}

/////
fn intersectTri4(
    ray: Ray,
    iTriangleIndex: u32) -> RayTriangleIntersectionResult
{
    //let iIndex0: u32 = aiDynamicTriangleIndices[iTriangleIndex * 3];
    //let iIndex1: u32 = aiDynamicTriangleIndices[iTriangleIndex * 3 + 1];
    //let iIndex2: u32 = aiDynamicTriangleIndices[iTriangleIndex * 3 + 2];

    let pos0: vec4<f32> = aDynamicVertexPositions[aiDynamicTriangleIndices[iTriangleIndex * 3]];
    let pos1: vec4<f32> = aDynamicVertexPositions[aiDynamicTriangleIndices[iTriangleIndex * 3 + 1]];
    let pos2: vec4<f32> = aDynamicVertexPositions[aiDynamicTriangleIndices[iTriangleIndex * 3 + 2]];

    var iIntersected: u32 = 0;
    var fT: f32 = FLT_MAX;
    let intersectionInfo: RayTriangleIntersectionResult = rayTriangleIntersection(
        ray.mOrigin.xyz,
        ray.mOrigin.xyz + ray.mDirection.xyz * 1000.0f,
        pos0.xyz,
        pos1.xyz,
        pos2.xyz);
    
    return intersectionInfo;
}

/////
fn intersectBVH4(
    ray: Ray,
    iRootNodeIndex: u32) -> IntersectBVHResult
{
    var ret: IntersectBVHResult;

    var iStackTop: i32 = 0;
    var aiStack: array<u32, 128>;
    aiStack[iStackTop] = iRootNodeIndex;

    ret.mHitPosition = vec3<f32>(FLT_MAX, FLT_MAX, FLT_MAX);
    ret.miHitTriangle = UINT32_MAX;
    var fClosestDistance: f32 = FLT_MAX;

    for(var iStep: u32 = 0u; iStep < 10000u; iStep++)
    {
        if(iStackTop < 0)
        {
            break;
        }

        let iNodeIndex: u32 = aiStack[iStackTop];
        iStackTop -= 1;

        //let node: BVHNode2 = aDynamicBVHNodes[iNodeIndex];
        if(aDynamicBVHNodes[iNodeIndex].miPrimitiveID != UINT32_MAX)
        {
            let intersectionInfo: RayTriangleIntersectionResult = intersectTri4(
                ray,
                aDynamicBVHNodes[iNodeIndex].miPrimitiveID);

            if(abs(intersectionInfo.mIntersectPosition.x) < 10000.0f)
            {
                let fDistanceToEye: f32 = length(intersectionInfo.mIntersectPosition.xyz - ray.mOrigin.xyz);
                //if(fDistanceToEye < fClosestDistance)
                {
                    
                    //fClosestDistance = fDistanceToEye;
                    ret.mHitPosition = intersectionInfo.mIntersectPosition.xyz;
                    ret.mHitNormal = intersectionInfo.mIntersectNormal.xyz;
                    ret.miHitTriangle = aDynamicBVHNodes[iNodeIndex].miPrimitiveID;
                    ret.mBarycentricCoordinate = intersectionInfo.mBarycentricCoordinate;

                    break;
                }
            }
        }
        else
        {
            let bIntersect: bool = rayBoxIntersect(
                ray.mOrigin.xyz,
                ray.mDirection.xyz,
                aDynamicBVHNodes[iNodeIndex].mMinBound.xyz,
                aDynamicBVHNodes[iNodeIndex].mMaxBound.xyz);

            // node left and right child to stack
            if(bIntersect)
            {
                iStackTop += 1;
                aiStack[iStackTop] = aDynamicBVHNodes[iNodeIndex].miChildren0;
                iStackTop += 1;
                aiStack[iStackTop] = aDynamicBVHNodes[iNodeIndex].miChildren1;
            }
        }
    }

    return ret;
}

/////
fn getPreviousScreenUV(
    screenUV: vec2<f32>) -> vec2<f32>
{
    var screenUVCopy: vec2<f32> = screenUV;
    var motionVector: vec2<f32> = textureSample(
        motionVectorTexture,
        textureSampler,
        screenUVCopy).xy;
    motionVector = motionVector * 2.0f - 1.0f;
    var prevScreenUV: vec2<f32> = screenUVCopy - motionVector;

    var worldPosition: vec3<f32> = textureSample(
        worldPositionTexture,
        textureSampler,
        screenUVCopy
    ).xyz;
    var normal: vec3<f32> = textureSample(
        normalTexture,
        textureSampler,
        screenUVCopy
    ).xyz;

    var fOneOverScreenWidth: f32 = 1.0f / f32(defaultUniformData.miScreenWidth);
    var fOneOverScreenHeight: f32 = 1.0f / f32(defaultUniformData.miScreenHeight);

    var fShortestWorldDistance: f32 = FLT_MAX;
    var closestScreenUV: vec2<f32> = prevScreenUV;
    for(var iY: i32 = -1; iY <= 1; iY++)
    {
        for(var iX: i32 = -1; iX <= 1; iX++)
        {
            var sampleUV: vec2<f32> = prevScreenUV + vec2<f32>(
                f32(iX) * fOneOverScreenWidth,
                f32(iY) * fOneOverScreenHeight 
            );

            sampleUV.x = clamp(sampleUV.x, 0.0f, 1.0f);
            sampleUV.y = clamp(sampleUV.y, 0.0f, 1.0f);

            var checkWorldPosition: vec3<f32> = textureSample(
                prevWorldPositionTexture,
                textureSampler,
                vec2<f32>(
                    clamp(sampleUV.x, 0.0f, 1.0f),
                    clamp(sampleUV.y, 0.0f, 1.0f)
                )
            ).xyz;
            var checkNormal: vec3<f32> = textureSample(
                prevNormalTexture,
                textureSampler,
                sampleUV
            ).xyz;
            var fNormalDP: f32 = abs(dot(checkNormal, normal));

            var worldPositionDiff: vec3<f32> = checkWorldPosition - worldPosition;
            var fLengthSquared: f32 = dot(worldPositionDiff, worldPositionDiff);
            if(fNormalDP >= 0.99f && fShortestWorldDistance > fLengthSquared)
            {
                fShortestWorldDistance = fLengthSquared;
                closestScreenUV = sampleUV;
            }
        }
    }

    return closestScreenUV;
}

/////
fn isDisoccluded2(
    screenUV: vec2<f32>,
    prevScreenUV: vec2<f32>
) -> bool
{
    var worldPosition: vec3<f32> = textureSample(
        worldPositionTexture,
        textureSampler,
        screenUV).xyz;

    var prevWorldPosition: vec3<f32> = textureSample(
        prevWorldPositionTexture,
        textureSampler,
        prevScreenUV).xyz;

    var normal: vec3<f32> = textureSample(
        normalTexture,
        textureSampler,
        screenUV).xyz;

    var prevNormal: vec3<f32> = textureSample(
        prevNormalTexture,
        textureSampler,
        prevScreenUV).xyz;

    var motionVector: vec4<f32> = textureSample(
        motionVectorTexture,
        textureSampler,
        screenUV);

    var prevMotionVectorAndMeshIDAndDepth: vec4<f32> = textureSample(
        prevMotionVectorTexture,
        textureSampler,
        prevScreenUV);

    let iMesh = u32(ceil(motionVector.z - 0.5f)) - 1;
    var fDepth: f32 = motionVector.w;
    var fPrevDepth: f32 = prevMotionVectorAndMeshIDAndDepth.w;
    var fCheckDepth: f32 = abs(fDepth - fPrevDepth);
    var worldPositionDiff: vec3<f32> = prevWorldPosition.xyz - worldPosition.xyz;
    var fCheckDP: f32 = abs(dot(normalize(normal.xyz), normalize(prevNormal.xyz)));
    let iPrevMesh: u32 = u32(ceil(prevMotionVectorAndMeshIDAndDepth.z - 0.5f)) - 1;
    var fCheckWorldPositionDistance: f32 = dot(worldPositionDiff, worldPositionDiff);

    return !(iMesh == iPrevMesh && fCheckDepth <= 0.01f && fCheckWorldPositionDistance <= 0.01f && fCheckDP >= 0.99f);
}

///// 
fn isPrevUVOutOfBounds(inputTexCoord: vec2<f32>) -> bool
{
    var motionVector: vec4<f32> = textureSample(
        motionVectorTexture,
        textureSampler,
        inputTexCoord);
    motionVector.x = motionVector.x * 2.0f - 1.0f;
    motionVector.y = motionVector.y * 2.0f - 1.0f;
    let backProjectedScreenUV: vec2<f32> = inputTexCoord - motionVector.xy;

    return (backProjectedScreenUV.x < 0.0f || backProjectedScreenUV.x > 1.0 || backProjectedScreenUV.y < 0.0f || backProjectedScreenUV.y > 1.0f);
}

/////
fn ToneMapFilmic_Hejl2015(
    hdr: vec3<f32>, 
    whitePt: f32) -> vec3<f32>
{
    let vh: vec4<f32> = vec4<f32>(hdr, whitePt);
    let va: vec4<f32> = (vh * 1.425f) + 0.05f;
    let vf: vec4<f32> = ((vh * va + 0.004f) / ((vh * (va + 0.55f) + 0.0491f))) - vec4<f32>(0.0821f);
    var ret: vec3<f32> = vf.rgb / vf.w;

    return clamp(ret, vec3<f32>(0.0f, 0.0f, 0.0f), vec3<f32>(1.0f, 1.0f, 1.0f));
}

/////
fn convertToSRGB(
    radiance: vec3<f32>) -> vec3<f32>
{
    let maxComp: f32 = max(max(radiance.x, radiance.y), radiance.z);
    let maxRadiance: vec3<f32> = max(radiance, 
        vec3<f32>(0.01f * maxComp));
    let linearRadiance: vec3<f32> = ACESFilm(maxRadiance);

    return linearToSRGB(linearRadiance);
}

/////
fn ACESFilm(
    radiance: vec3<f32>) -> vec3<f32>
{
    let fA: f32 = 2.51f;
    let fB: f32 = 0.03f;
    let fC: f32 = 2.43f;
    let fD: f32 = 0.59f;
    let fE: f32 = 0.14f;

    return saturate((radiance * (fA * radiance + fB)) / (radiance * (fC * radiance + fD) + fE));
}

/////
fn linearToSRGB(
    x: vec3<f32>) -> vec3<f32>
{
    var bCond: bool = (x.x < 0.0031308f || x.y < 0.0031308f || x.z < 0.0031308f);
    var ret: vec3<f32> = x * 12.92f;
    if(!bCond) 
    {
        ret = vec3<f32>(
            pow(x.x, 1.0f / 2.4f) * 1.055f - 0.055f,
            pow(x.y, 1.0f / 2.4f) * 1.055f - 0.055f,
            pow(x.z, 1.0f / 2.4f) * 1.055f - 0.055f
        );
    }

    return ret;
}