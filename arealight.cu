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

rtBuffer<float, 1>              gaussian_lookup;

__device__ __inline__ float gaussFilter(float distsq, float scale)
{
  float sample = distsq/(scale*scale);
  if (sample > 0.9999) {
    return 0.0;
  }
  float scaled = sample*64;
  int index = (int) scaled;
  float weight = scaled - index;
  return (1.0 - weight) * gaussian_lookup[index] + weight * gaussian_lookup[index + 1];
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
rtBuffer<float3, 2>               world_loc;
rtBuffer<float3, 2>               n;
rtBuffer<int, 2>                  err_buf;
rtBuffer<float, 2>                dist_scale;
rtDeclareVariable(uint,           frame, , );
rtDeclareVariable(uint,           blur_occ, , );
rtDeclareVariable(uint,           err_vis, , );
rtDeclareVariable(uint,						view_zmin, , );

rtDeclareVariable(uint,           normal_rpp, , );
rtDeclareVariable(uint,           brute_rpp, , );
rtDeclareVariable(uint,           show_progressive, , );
rtDeclareVariable(float,          zmin_rpp_scale, , );
rtDeclareVariable(int2,           pixel_radius, , );


rtDeclareVariable(uint,           show_brdf, , );
rtDeclareVariable(uint,           show_occ, , );

rtBuffer<uint, 2>                 conv;

RT_PROGRAM void pinhole_camera() {

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
    err_buf[launch_index] = 0;
  }


  float4 cur_occ = occ[launch_index];
  float scale = cur_occ.z;

  if (scale < 0) {
    output_buffer[launch_index] = make_color(bg_color);

    return;
  }


  if (frame == 1 && scale < 0.05 && cur_occ.x>0.01) {
    prd.sqrt_num_samples = brute_rpp;
    prd.brdf = true;
    //shoot_ray = true;
    shoot_ray = false;
    err_buf[launch_index] = 1;
  }


  float3 cur_world_loc = make_float3(0.0);

  if (shoot_ray) {
    optix::Ray ray(ray_origin, ray_direction, radiance_ray_type, scene_epsilon );

    prd.importance = 1.f;
    prd.unavg_occ = 0;
    prd.depth = 0;
    prd.hit = false;

    rtTrace(top_object, ray, prd);

    if(!prd.hit) {
      occ[launch_index] = make_float4(1,1,-1,0);
      output_buffer[launch_index] = make_color(bg_color);
      return;
    }

    cur_world_loc = prd.world_loc;
    world_loc[launch_index] = cur_world_loc;
    n[launch_index] = normalize(prd.n);

    dist_scale[launch_index] = prd.dist_scale;


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

  //i guess just blur here for now... inefficient, but gets the point across
  if (blur_occ && (frame > 1)) {
    int numBlurred = 0;

    float3 cur_n = n[make_uint2(launch_index.x, launch_index.y)];

    if (scale > 0.0) {
      for(int i=-pixel_radius.x; i < pixel_radius.x; i++) {
        for(int j=-pixel_radius.y; j < pixel_radius.y; j++) {
          if(launch_index.x + i > 0 && launch_index.y + j > 0) {
            uint2 target_index = make_uint2(launch_index.x+i, launch_index.y+j);
            if(target_index.x < output_buffer.size().x && target_index.y < output_buffer.size().y && occ[target_index].z > 0) {
              float target_occ = occ[make_uint2(launch_index.x+i, launch_index.y+j)].x;
              float3 target_n = n[make_uint2(launch_index.x+i, launch_index.y+j)];
              //float distance = target_occ.w - prd.t_hit;
              float3 loca = cur_world_loc;
              float3 locb = world_loc[make_uint2(launch_index.x+i, launch_index.y+j)];
              float3 diff = loca-locb;
              float distancesq = diff.x*diff.x + diff.y*diff.y + diff.z*diff.z;
              if(distancesq < 0)
                distancesq = -distancesq;
              if (distancesq < 1) {
                if (acos(dot(target_n, cur_n)) < 0.785) {
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
    }
    if(sumWeight > 0)
      blurred_occ /= sumWeight;
  } else {
    blurred_occ = occ[launch_index].x;
  }


  //brdf info in brdf[launch_index], not yet computed correctly, to save time.
  //output_buffer[launch_index] = make_color( make_float3(blurred_occ));
  float3 brdf_term = make_float3(1);
  float occ_term = 1;
  if (show_brdf)
    brdf_term = brdf[launch_index];
  if (show_occ)
	  occ_term = blurred_occ;
  output_buffer[launch_index] = make_color( occ_term * brdf_term);
  if (view_zmin)
    output_buffer[launch_index] = make_color( make_float3(scale) );
/*
  if (shoot_ray && err_vis)
    output_buffer[launch_index] = make_color( make_float3(0,1,0) );
*/
  if(err_vis) {
    int cur_err = err_buf[launch_index];
    if(cur_err != 0)
      output_buffer[launch_index] = make_color ( make_float3(cur_err==1, cur_err==2, cur_err==3) );
  }
/*
  if (err_vis)
    output_buffer[launch_index] = make_color( make_float3(0,0,(float)numIgnored/10) );
*/
  //output_buffer[launch_index] = make_color(make_float3(dist_scale[launch_index]));
}


//
// Returns solid color for miss rays
//
RT_PROGRAM void miss()
{
  prd_radiance.result = bg_color;
  //prd_radiance.shadow_intersection = 100000.0f;
}

//
// Terminates and fully attenuates ray after any hit
//
RT_PROGRAM void any_hit_shadow()

{
  // this material is opaque, so it fully attenuates all shadow rays
  prd_shadow.attenuation = make_float3(0);
  prd_shadow.distance = t_hit;

  rtTerminateRay();
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

  //hardcoded fov i guess
  float fov = 60.0;
  float z_dist = (t_hit*ray.direction).z;
  prd_radiance.dist_scale = z_dist*tan(30.0*M_PI/180.0)/360.0;


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

  //Stratify x
  int num_occ = 0;
  for(int i=0; i<prd_radiance.sqrt_num_samples; ++i) {
    seed.x = rot_seed(seed.x, i);

    //Stratify y
    for(int j=0; j<prd_radiance.sqrt_num_samples; ++j) {
      seed.y = rot_seed(seed.y, j);

      float2 sample = make_float2( rnd(seed.x), rnd(seed.y) );
      sample.x = (sample.x+((float)i))/prd_radiance.sqrt_num_samples;
      sample.y = (sample.y+((float)j))/prd_radiance.sqrt_num_samples;

      /*
      //From point, choose a random direction to sample in
      float3 U, V, W;
      float3 sampleDir;
      createONB( ffnormal, U, V, W); //(is ffnormal the correct one to be using here?)
      sampleUnitHemisphere( sample, U, V, W, sampleDir );
       */

      float3 target = (sample.x * lx + sample.y * ly) + lo;
      float3 sampleDir = normalize(target - hit_point);

      if(dot(ffnormal, sampleDir) > 0.0f) {
        ++num_occ;

        // PHONG

        float3 L = normalize(target - hit_point);
        float nDl = dot( ffnormal, L);
        float3 H = normalize(sampleDir - ray.direction);
        float nDh = dot( ffnormal, H );
        //temporary - white light
        float3 Lc = make_float3(1,1,1);
        colorAvg += Kd * nDl * Lc;
        if (nDh > 0)
          colorAvg += Ks * pow(nDh, phong_exp);



		float3 distancevectoocc = target-hit_point;
		float distancetolight = sqrt(distancevectoocc.x*distancevectoocc.x + distancevectoocc.y*distancevectoocc.y + distancevectoocc.z * distancevectoocc.z);


        // SHADOW
        //cast ray and check for shadow
        PerRayData_shadow shadow_prd;
        shadow_prd.attenuation = make_float3(1.0f);
        shadow_prd.distance = distancetolight;
        optix::Ray shadow_ray ( hit_point, sampleDir, shadow_ray_type, 0.001);//scene_epsilon );
        rtTrace(top_shadower, shadow_ray, shadow_prd);
        occlusion += shadow_prd.attenuation.x;

				float scale = distancetolight/(distancetolight-shadow_prd.distance) - 1;
        prd_radiance.gauss_scale = min(scale, prd_radiance.gauss_scale);
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
  color += colorAvg/num_occ;
  shadow_rng_seeds[launch_index] = seed;


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
