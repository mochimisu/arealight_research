/*
 * arealight.cu
 * Area Light Filtering
 * Adapted from NVIDIA OptiX Tutorial
 * Brandon Wang, Soham Mehta
 */

#include "arealight.h"

rtDeclareVariable(float3, geometric_normal, attribute geometric_normal, );
rtDeclareVariable(float3, shading_normal,   attribute shading_normal, );

rtDeclareVariable(PerRayData_radiance, prd_radiance, rtPayload, );
rtDeclareVariable(PerRayData_shadow,   prd_shadow,   rtPayload, );

rtDeclareVariable(optix::Ray, ray,          rtCurrentRay, );
rtDeclareVariable(float,      t_hit,        rtIntersectionDistance, );
rtDeclareVariable(uint2,      launch_index, rtLaunchIndex, );

rtDeclareVariable(unsigned int, radiance_ray_type, , );
rtDeclareVariable(unsigned int, shadow_ray_type , , );
rtDeclareVariable(float,        scene_epsilon, , );
rtDeclareVariable(rtObject,     top_object, , );

rtDeclareVariable(float,          light_sigma, , );

// Create ONB from normal.  Resulting W is Parallel to normal
__device__ __inline__ void createONB( const optix::float3& n,
    optix::float3& U,
    optix::float3& V,
    optix::float3& W )
{
  using namespace optix;

  W = normalize( n );
  U = cross( W, make_float3( 0.0f, 1.0f, 0.0f ) );
  if ( fabsf( U.x) < 0.001f && fabsf( U.y ) < 0.001f && fabsf( U.z ) < 0.001f  )
    U = cross( W, make_float3( 1.0f, 0.0f, 0.0f ) );
  U = normalize( U );
  V = cross( W, U );
}

// Create ONB from normalalized vector
__device__ __inline__ void createONB( const optix::float3& n,
    optix::float3& U,
    optix::float3& V)
{
  using namespace optix;
  U = cross( n, make_float3( 0.0f, 1.0f, 0.0f ) );
  if ( dot(U, U) < 1.e-3f )
    U = cross( n, make_float3( 1.0f, 0.0f, 0.0f ) );
  U = normalize( U );
  V = cross( n, U );
}

// sample hemisphere with cosine density
__device__ __inline__ void sampleUnitHemisphere( const optix::float2& sample,
    const optix::float3& U,
    const optix::float3& V,
    const optix::float3& W,
    optix::float3& point )
{
  using namespace optix;

  float phi = 2.0f * M_PIf*sample.x;
  float r = sqrt( sample.y );
  float x = r * cos(phi);
  float y = r * sin(phi);
  float z = 1.0f - x*x -y*y;
  z = z > 0.0f ? sqrt(z) : 0.0f;

  point = x*U + y*V + z*W;
}

__device__ __inline__ float3 heatMap(float val) {
  float fraction;
  if (val < 0.0f)
    fraction = -1.0f;
  else if (val > 1.0f)
    fraction = 1.0f;
  else
    fraction = 2.0f * val - 1.0f;

  if (fraction < -0.5f)
    return make_float3(0.0f, 2*(fraction+1.0f), 1.0f);
  else if (fraction < 0.0f)
    return make_float3(0.0f, 1.0f, 1.0f - 2.0f * (fraction + 0.5f));
  else if (fraction < 0.5f)
    return make_float3(2.0f * fraction, 1.0f, 0.0f);
  else
    return make_float3(1.0f, 1.0f - 2.0f*(fraction - 0.5f), 0.0f);
}

rtBuffer<float, 1>              gaussian_lookup;

__device__ __inline__ float gaussFilter(float distsq, float scale)
{
  float sample = distsq/(scale*scale * light_sigma * light_sigma);
  if (sample > 0.9999) {
    return 0.0;
  }
  float scaled = sample*64;
  int index = (int) scaled;
  float weight = scaled - index;
  return (1.0 - weight) * gaussian_lookup[index] + weight * gaussian_lookup[index + 1];
}

//marsaglia polar method
__device__ __inline__ float2 randomGauss(float center, float std_dev, float2 sample)
{
  float u,v,s;
  u = sample.x * 2 - 1;
  v = sample.y * 2 - 1;
  s = u*u + v*v;
  float2 result = make_float2(
      center+std_dev*v*sqrt(-2.0*log(s)/s),
      center+std_dev*u*sqrt(-2.0*log(s)/s));
  return result;
}

//
// Pinhole camera implementation
//
rtDeclareVariable(float3,        eye, , );
rtDeclareVariable(float3,        U, , );
rtDeclareVariable(float3,        V, , );
rtDeclareVariable(float3,        W, , );
rtDeclareVariable(float3,        bad_color, , );
rtBuffer<uchar4, 2>              output_buffer;

rtDeclareVariable(float3, bg_color, , );

rtBuffer<float3, 2>               brdf;
//divided occlusion, undivided occlusion, zmin, num_samples
rtBuffer<float4, 2>               occ;
rtBuffer<float, 2>                occ_blur1d;
rtBuffer<float3, 2>               world_loc;
rtBuffer<float3, 2>               n;
rtBuffer<float, 2>                dist_scale;
rtBuffer<float2, 2>               zdist;
rtBuffer<float, 2>                spp;
rtBuffer<float, 2>                spp_cur;
//s1,s2
rtBuffer<float2, 2>               slope;
rtDeclareVariable(uint,           frame, , );
rtDeclareVariable(uint,           blur_occ, , );
rtDeclareVariable(uint,           err_vis, , );
rtDeclareVariable(uint,						view_mode, , );

rtDeclareVariable(uint,           normal_rpp, , );
rtDeclareVariable(uint,           brute_rpp, , );
rtDeclareVariable(uint,           show_progressive, , );
rtDeclareVariable(float,          zmin_rpp_scale, , );
rtDeclareVariable(int2,           pixel_radius, , );

rtDeclareVariable(uint,           show_brdf, , );
rtDeclareVariable(uint,           show_occ, , );

RT_PROGRAM void pinhole_camera() {
  int cur_err = 0;

  size_t2 screen = output_buffer.size();

  float2 d = make_float2(launch_index) / make_float2(screen) * 2.f - 1.f;
  float3 ray_origin = eye;
  float3 ray_direction = normalize(d.x*U + d.y*V + W);
  PerRayData_radiance prd;

  bool shoot_ray = false;
  if (frame == 0) {
    occ[launch_index] = make_float4(1.0, 1.0, 100000.0, 0.0);
    prd.sqrt_num_samples = normal_rpp;
    prd.brdf = true;
    shoot_ray = true;
    zdist[launch_index] = make_float2(10000.0,0);
    slope[launch_index] = make_float2(0,10000.0);
    spp[launch_index] = normal_rpp * normal_rpp;
    spp_cur[launch_index] = normal_rpp * normal_rpp;
  }


  float4 cur_occ = occ[launch_index];
  float scale = cur_occ.z;

  if (scale < 0) {
    output_buffer[launch_index] = make_color(bg_color);
    spp[launch_index] = 0;
    return;
  }


  //if(frame>=1) {
  
  if (frame >= 1 && spp_cur[launch_index] < spp[launch_index]) {
    int target_samp = ceil(spp[launch_index]);
    int new_samp = max((int)ceil(target_samp - spp_cur[launch_index]), 1);
    int sqrt_samp = min(ceil(sqrt((float)new_samp)),4.0);
    prd.sqrt_num_samples = sqrt_samp;
    spp_cur[launch_index] = spp_cur[launch_index]+sqrt_samp*sqrt_samp;
    //prd.sqrt_num_samples = 1;
    //spp_cur[launch_index] = spp_cur[launch_index]+1;
    //prd.sqrt_num_samples = 7;
    //spp_cur[launch_index] = spp_cur[launch_index]+49;
    prd.brdf = false;
    shoot_ray = true;
    //shoot_ray = false;
    cur_err = 1;
  }

  if (spp[launch_index] < 0) 
    cur_err = 2;
  if (spp[launch_index] > 100000000)
    cur_err = 3;
  
  /*

  if (frame >= 1) {
    prd.brdf = false;
    shoot_ray = true;
    int sqrt_samp = 5;
    prd.sqrt_num_samples = sqrt_samp;
    spp_cur[launch_index] = spp_cur[launch_index]+sqrt_samp*sqrt_samp;
  }
  */


  float3 cur_world_loc = make_float3(0.0);

  if (shoot_ray) {
    optix::Ray ray(ray_origin, ray_direction, radiance_ray_type, scene_epsilon );

    prd.importance = 1.f;
    prd.unavg_occ = 0;
    prd.depth = 0;
    prd.hit = false;
    prd.d2min = zdist[launch_index].x;
    prd.d2max = zdist[launch_index].y;
    prd.s1 = slope[launch_index].x;
    prd.s2 = slope[launch_index].y;

    rtTrace(top_object, ray, prd);

    if(!prd.hit) {
      occ[launch_index] = make_float4(1,1,-1,0);
      output_buffer[launch_index] = make_color(bg_color);
      spp[launch_index] = 0;
      return;
    }

    cur_world_loc = prd.world_loc;
    world_loc[launch_index] = cur_world_loc;
    n[launch_index] = normalize(prd.n);

    dist_scale[launch_index] = prd.dist_scale;
    zdist[launch_index] = make_float2(min(prd.d2min, zdist[launch_index].x), 
        max(prd.d2max, zdist[launch_index].y));
    slope[launch_index] = make_float2(max(slope[launch_index].x, prd.s1),
        min(slope[launch_index].y, prd.s2));
    spp[launch_index] = prd.spp;// min(prd.spp,4000.0);

    if (prd.brdf)
      brdf[launch_index] = prd.result;

    if (cur_occ.z == 0) {
      float num_samples = prd.sqrt_num_samples * prd.sqrt_num_samples;
      scale = prd.gauss_scale;
      cur_occ = make_float4(prd.unavg_occ / num_samples, prd.unavg_occ, prd.gauss_scale, num_samples);
      occ[launch_index] = cur_occ;

    }
    else {
      //float precision loss maybe... but eh
      float total_undiv_occ = prd.unavg_occ + cur_occ.y;
      float num_samples = prd.sqrt_num_samples * prd.sqrt_num_samples + cur_occ.w;
      scale = min(prd.gauss_scale, scale);
      cur_occ = make_float4(total_undiv_occ / num_samples, total_undiv_occ, scale, num_samples);
      occ[launch_index] = cur_occ;

    }
  } else {
    cur_world_loc = world_loc[launch_index];
  }

  float blurred_occ = 0.0;
  float sumWeight = 0.0;
  float first_blurred_occ = 0.0;

  int pix_r_scale = floor(dist_scale[launch_index]*20)+1;

  //scale = 1.0;

  float occ_epsilon = 0.00001f;

  //if(occ[launch_index].x > 1.0 - occ_epsilon) 
  //  cur_err = 2;

  float dist_scale_threshold = 10000000000000000.0f;

  if (blur_occ && (frame > 1)) {// && occ[launch_index].x < 1.0-occ_epsilon ) {
    int numBlurred = 0;

    float3 cur_n = n[make_uint2(launch_index.x, launch_index.y)];

    //Testing out distance scale
    //int2 active_pixel_radius = make_int2(pix_r_scale, pix_r_scale);
    int2 active_pixel_radius = pixel_radius;
    if (scale > 0.0) {
      for(int i=-active_pixel_radius.x; i < active_pixel_radius.x; i++) {
        int j = 0; 
          if(launch_index.x + i > 0 && launch_index.y + j > 0) {
            uint2 target_index = make_uint2(launch_index.x+i, launch_index.y+j);
            if(target_index.x < output_buffer.size().x && target_index.y < output_buffer.size().y && occ[target_index].z > 0 
              && abs(occ[launch_index].z-occ[target_index].z) < dist_scale_threshold) {
              //float distance = target_occ.w - prd.t_hit;
              float3 loca = cur_world_loc;
              float3 locb = world_loc[make_uint2(launch_index.x+i, launch_index.y+j)];
              float3 diff = loca-locb;
              float distancesq = diff.x*diff.x + diff.y*diff.y + diff.z*diff.z;
              if (distancesq < 0)
                distancesq = -distancesq;
              if (distancesq < 1) {
                float3 target_n = n[make_uint2(launch_index.x+i, launch_index.y+j)];
                if (acos(dot(target_n, cur_n)) < 0.785) {
                  float target_occ = occ[make_uint2(launch_index.x+i, launch_index.y+j)].x;
                  //scale = (distance to light/distance to occluder - 1)
                  float weight = gaussFilter(distancesq,scale);
                  first_blurred_occ += weight * target_occ;
                  sumWeight += weight;
                  if (weight > 0)
                    numBlurred += 1;
                }
              }
            }
          }
      }
    }
    if(sumWeight > 0)
      first_blurred_occ /= sumWeight;

  } else {
    first_blurred_occ = occ[launch_index].x;
  }

  if (blur_occ && (frame > 2) ) { // && occ[launch_index].x < 1.0-occ_epsilon) {
    sumWeight = 0.0;
    int numBlurred = 0;

    float3 cur_n = n[make_uint2(launch_index.x, launch_index.y)];

    //Testing out distance scale
    int2 active_pixel_radius = make_int2(pix_r_scale, pix_r_scale);
    active_pixel_radius = pixel_radius;
    if (scale > 0.0) {
        for(int j=-active_pixel_radius.y; j < active_pixel_radius.y; j++) {
        int i = 0; 
          if(launch_index.x + i > 0 && launch_index.y + j > 0) {
            uint2 target_index = make_uint2(launch_index.x+i, launch_index.y+j);
            if(target_index.x < output_buffer.size().x && target_index.y < output_buffer.size().y && occ[target_index].z > 0
              && abs(occ[launch_index].z-occ[target_index].z) < dist_scale_threshold) {
              //float distance = target_occ.w - prd.t_hit;
              float3 loca = cur_world_loc;
              float3 locb = world_loc[make_uint2(launch_index.x+i, launch_index.y+j)];
              float3 diff = loca-locb;
              float distancesq = diff.x*diff.x + diff.y*diff.y + diff.z*diff.z;
              if(distancesq < 0)
                distancesq = -distancesq;
              if (distancesq < 1) {
                float3 target_n = n[make_uint2(launch_index.x+i, launch_index.y+j)];
                if (acos(dot(target_n, cur_n)) < 0.785) {
                  float target_occ = occ_blur1d[make_uint2(launch_index.x+i, launch_index.y+j)];
                  //scale = (distance to light/distance to occluder - 1)
                  float weight = gaussFilter(distancesq,scale);
                  blurred_occ += weight * target_occ;
                  sumWeight += weight;
                  if (weight > 0)
                    numBlurred += 1;
                }
              }
            }
        }
      }
    }
    if(sumWeight > 0)
      blurred_occ /= sumWeight;

  } else {
    blurred_occ = occ[launch_index].x;
  }

  occ_blur1d[launch_index] = first_blurred_occ;


  //brdf info in brdf[launch_index], not yet computed correctly, to save time.
  float3 brdf_term = make_float3(1);
  float occ_term = 1;
  if (show_brdf)
    brdf_term = brdf[launch_index];
  if (show_occ)
    occ_term = blurred_occ;
  if (view_mode) {
    if (view_mode == 1)
      //Occlusion only
      output_buffer[launch_index] = make_color( make_float3(blurred_occ) );
    if (view_mode == 2) 
      //Scale
      //output_buffer[launch_index] = make_color( make_float3(scale) );
      output_buffer[launch_index] = make_color( heatMap(scale) );
    if (view_mode == 3) 
      //Zmin
      if(shoot_ray)
      //output_buffer[launch_index] = make_color( make_float3(prd.d2min) / 100.0);
      output_buffer[launch_index] = make_color( heatMap(prd.d2min / 100.0) );
    if (view_mode == 4) {
      //Zmax
      if (shoot_ray) {
        if (prd.d2max > 0.001)
          //output_buffer[launch_index] = make_color( make_float3(prd.d2max) / 100.0);
          output_buffer[launch_index] = make_color( heatMap( prd.d2max / 100.0) );
        else
          output_buffer[launch_index] = make_color( heatMap( 1.0f ) );//make_float3(1) );
      }
    }
    if (view_mode == 5) 
      //Current SPP
      //output_buffer[launch_index] = make_color( make_float3(spp_cur[launch_index]) / 100.0 );
      output_buffer[launch_index] = make_color( heatMap(spp_cur[launch_index] / 100.0 ) );
    if (view_mode == 6) 
      //Theoretical SPP
      //output_buffer[launch_index] = make_color( make_float3(spp[launch_index]) / 100.0 );
      output_buffer[launch_index] = make_color( heatMap(spp[launch_index] / 100.0 ) );
  } else
    output_buffer[launch_index] = make_color( occ_term * brdf_term);

  
  if(err_vis) {
    if(cur_err != 0)
      output_buffer[launch_index] = make_color ( make_float3(cur_err==1, cur_err==2, cur_err==3) );
  }
  
}


//
// Returns solid color for miss rays
//
RT_PROGRAM void miss()
{
  prd_radiance.result = bg_color;
  prd_radiance.spp = 0;
  //prd_radiance.shadow_intersection = 100000.0f;
}

//
// Terminates and fully attenuates ray after any hit
//
RT_PROGRAM void any_hit_shadow()

{
  // this material is opaque, so it fully attenuates all shadow rays
  prd_shadow.attenuation = make_float3(0);
  prd_shadow.hit = true;
  //prd_shadow.distance = t_hit;
  
  prd_shadow.distance_min = min(prd_shadow.distance_min, t_hit);
  prd_shadow.distance_max = max(prd_shadow.distance_max, t_hit);

  rtIgnoreIntersection();

  //rtTerminateRay();
}


//
// Phong surface shading with shadows
//
rtDeclareVariable(float3,   Ka, , );
rtDeclareVariable(float3,   Ks, , );
rtDeclareVariable(float,    phong_exp, , );
rtDeclareVariable(float3,   Kd, , );
rtDeclareVariable(float3,   ambient_light_color, , );
rtBuffer<AreaLight>        lights;
rtDeclareVariable(rtObject, top_shadower, , );
rtDeclareVariable(float3, reflectivity, , );
rtDeclareVariable(float, importance_cutoff, , );
rtDeclareVariable(int, max_depth, , );

//asdf
rtBuffer<uint2, 2> shadow_rng_seeds;

RT_PROGRAM void closest_hit_radiance3()
{
  float3 world_geo_normal   = normalize( rtTransformNormal( RT_OBJECT_TO_WORLD, geometric_normal ) );
  float3 world_shade_normal = normalize( rtTransformNormal( RT_OBJECT_TO_WORLD, shading_normal ) );
  float3 ffnormal     = faceforward( world_shade_normal, -ray.direction, world_geo_normal );
  float3 color = Ka * ambient_light_color;

  float3 hit_point = ray.origin + t_hit * ray.direction;
  prd_radiance.t_hit = t_hit;
  prd_radiance.world_loc = hit_point;
  prd_radiance.hit = true;
  prd_radiance.n = ffnormal;
  //occlusion values
  unsigned int occlusion = 0;

  prd_radiance.dist_scale = 1.0/(t_hit*tan(30.0*M_PI/180.0)); 
      //divide by 360 to get absolute between pixel and image


  uint2 seed = shadow_rng_seeds[launch_index];
  //seed.x = rot_seed(seed.x, frame);
  //seed.y = rot_seed(seed.y, frame);
  //float2 sample = make_float2( rnd(seed.x), rnd(seed.y) );

  //shadow_rng_seeds[launch_index] = seed;


  prd_radiance.gauss_scale = 1000000.0f;
  //Assume 1 light for now
  AreaLight light = lights[0];
  float3 lx = light.v2 - light.v1;
  float3 ly = light.v3 - light.v1;
  float3 lo = light.v1;
  //float3 lc = light.color;
  float3 colorAvg = make_float3(0,0,0);

  //phong values
  //assuming ambocc for now
  //THIS IS WRONG, FIX IT SOON
  /*
     if (prd_radiance.brdf) {

     float3 H = normalize(ffnormal - ray.direction);
     float nDh = dot( ffnormal, H );
     if (nDh > 0)
     colorAvg += Ks * pow(nDh, phong_exp);
     }
   */
  if(prd_radiance.brdf) {
    float3 phong_target = 0.5 * lx + 0.5 * ly + lo;
    float3 phong_dir = normalize( phong_target - hit_point );

    float3 L = normalize(phong_target - hit_point);
    float nDl = dot( ffnormal, L );
    float3 H = normalize(phong_dir - ray.direction);
    float nDh = dot( ffnormal, H );
    //temporary - white light
    float3 Lc = make_float3(1,1,1);
    color += Kd * nDl * Lc;// * strength;
    if (nDh > 0)
      color += Ks * pow(nDh, phong_exp);
  }

  //hardcoded sigma for now (for light intensity)

  //Stratify x
  float num_occ = 0;
  float occ_strength_tot = 0.0;
  float distance_summed = 0.0;
  for(int i=0; i<prd_radiance.sqrt_num_samples; ++i) {
    seed.x = rot_seed(seed.x, i);

    //Stratify y
    for(int j=0; j<prd_radiance.sqrt_num_samples; ++j) {
      seed.y = rot_seed(seed.y, j);

      float2 sample = make_float2( rnd(seed.x), rnd(seed.y) );
      sample.x = (sample.x+((float)i))/prd_radiance.sqrt_num_samples;
      sample.y = (sample.y+((float)j))/prd_radiance.sqrt_num_samples;

      //float strength = 1.0f;

      /*
      float strength = (exp(-(sample.x - 0.5) \
            * (sample.x - 0.5)/(2*light_sigma*light_sigma))) \
            * (1/(light_sigma * sqrt(M_2_PI)) * exp(-(sample.y - 0.5) \
            * (sample.y - 0.5)/(2*light_sigma*light_sigma)));
        */

      float strength = exp((((sample.x-0.5) * (sample.x-0.5)) \
        + ((sample.y - 0.5) * (sample.y - 0.5))) \
        / (2 * light_sigma * light_sigma));

      //what does this term do?
      //strength *= 1/(light_sigma * sqrt(M_2_PI));

      //it looks too strong or something
      //strength /= 3.0;

      /*
      //From point, choose a random direction to sample in
      float3 U, V, W;
      float3 sampleDir;
      createONB( ffnormal, U, V, W); //(is ffnormal the correct one to be using here?)
      sampleUnitHemisphere( sample, U, V, W, sampleDir );
       */

      float3 target = (sample.x * lx + sample.y * ly) + lo;
      float3 sampleDir = normalize(target - hit_point);

      float3 distancevectoocc = target-hit_point;
      float distancetolight = sqrt(distancevectoocc.x*distancevectoocc.x + distancevectoocc.y*distancevectoocc.y + distancevectoocc.z * distancevectoocc.z);
      distance_summed += distancetolight;

      if(dot(ffnormal, sampleDir) > 0.0f) {
        num_occ += strength;
        //++num_occ;
        //occ_strength_tot += strength;

        // PHONG
        /*
        float3 L = normalize(target - hit_point);
        float nDl = dot( ffnormal, L);
        float3 H = normalize(sampleDir - ray.direction);
        float nDh = dot( ffnormal, H );
        //temporary - white light
        float3 Lc = make_float3(1,1,1);
        colorAvg += Kd * nDl * Lc * strength;
        if (nDh > 0)
          colorAvg += Ks * pow(nDh, phong_exp);
          */

        // SHADOW
        //cast ray and check for shadow
        PerRayData_shadow shadow_prd;
        shadow_prd.attenuation = make_float3(strength);
        shadow_prd.distance_max = 0;
        shadow_prd.distance_min = distancetolight;
        shadow_prd.hit = false;
        optix::Ray shadow_ray ( hit_point, sampleDir, shadow_ray_type, 0.001);//scene_epsilon );
        rtTrace(top_shadower, shadow_ray, shadow_prd);
        occlusion += shadow_prd.attenuation.x;
 
        if(shadow_prd.hit) {
          float d2min = distancetolight - shadow_prd.distance_max;
          float d2max = distancetolight - shadow_prd.distance_min;
          if (shadow_prd.distance_max < 0.000000001)
            d2min = distancetolight;

          prd_radiance.d2min = d2min;
          prd_radiance.d2max = d2max;

          float scale = distancetolight/d2min - 1;
          prd_radiance.gauss_scale = min(scale, prd_radiance.gauss_scale);
        }
        /*
        float d2 = distancetolight - shadow_prd.distance;
        if(d2 > scene_epsilon) {

          //prd_radiance.d2min = d2;
          prd_radiance.d2min = min(prd_radiance.d2min, d2);
          //prd_radiance.d2max = d2;
          prd_radiance.d2max = max(prd_radiance.d2max, d2);
          float scale = distancetolight/(d2) - 1;
          prd_radiance.gauss_scale = min(scale, prd_radiance.gauss_scale);
        }*/
        //prd_radiance.shadow_intersection = min(dlzmin, prd_radiance.shadow_intersection);
        //prd_radiance.shadow_intersection = min(1/dlzmin, prd_radiance.shadow_intersection);
        //prd_radiance.shadow_intersection = min(dzmindl,prd_radiance.shadow_intersection);
        //prd_radiance.shadow_intersection = min(shadow_prd.distance,prd_radiance.shadow_intersection);

      }
      /*
         else {
         color += make_float3(10000,0,0);
         }
       */
    }
  }
  //color += colorAvg/(prd_radiance.sqrt_num_samples*prd_radiance.sqrt_num_samples);
  //color += colorAvg/occ_strength_tot;
  shadow_rng_seeds[launch_index] = seed;
  distance_summed /= prd_radiance.sqrt_num_samples * prd_radiance.sqrt_num_samples;

  /*
     float importance = prd_radiance.importance * optix::luminance( reflectivity );

     if( importance > importance_cutoff && prd_radiance.depth < max_depth) {
     PerRayData_radiance refl_prd;
     refl_prd.importance = importance;
     refl_prd.depth = prd_radiance.depth+1;
     float3 R = reflect( ray.direction, ffnormal );
     optix::Ray refl_ray( hit_point, R, radiance_ray_type, scene_epsilon );
     rtTrace(top_object, refl_ray, refl_prd);
  //color += reflectivity * refl_prd.result;
  }*/

  float s1 = distance_summed/prd_radiance.d2min - 1.0;
  float s2 = distance_summed/prd_radiance.d2max - 1.0;
  s1 = max(prd_radiance.s1,s1);
  s2 = min(prd_radiance.s2,s2);
  prd_radiance.s1 = s1;
  prd_radiance.s2 = s2;

  //float spp = 4.0*(1.0+s1/s2)*(1.0+s1/s2);

  /*
  float ap = 1.0/360.0 * 1.0/(t_hit*tan(30.0*M_PI/180.0)); 
  ap = ap*ap;

  float al = 36.0 * light_sigma * light_sigma;

  float omega_max_pix = 0.5 / (sqrt(ap));

  float omega_f_y = 2.0/light_sigma;
  float omega_f_x = 2.0/(light_sigma*s2);

  float omega_star_x = omega_f_x + omega_max_pix;
  float omega_star_y = omega_f_y + s1*omega_f_x;

  float spp = (omega_star_x * omega_star_y) * (omega_star_x * omega_star_y) *
    ap * al;
  */

  //assume d is same in all dim
  float d = 1.0/360.0 * (t_hit*tan(30.0*M_PI/180.0));
  float omega_l_max = 2.0/light_sigma;

  float spp_t_1 = (1+d*(omega_l_max)/s2);
  float spp_t_2 = (1+s1/s2);
  float spp = 4*spp_t_1*spp_t_1*spp_t_2*spp_t_2;
  
  
  prd_radiance.spp = spp;

  prd_radiance.unavg_occ = occlusion;
  prd_radiance.num_occ = num_occ;
  prd_radiance.result = color;
}


//
// Set pixel to solid color upon failure
//
RT_PROGRAM void exception()
{
  output_buffer[launch_index] = make_color( bad_color );
}
