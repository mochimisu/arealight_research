/*
 * Copyright (c) 2008 - 2009 NVIDIA Corporation.  All rights reserved.
 *
 * NVIDIA Corporation and its licensors retain all intellectual property and proprietary
 * rights in and to this software, related documentation and any modifications thereto.
 * Any use, reproduction, disclosure or distribution of this software and related
 * documentation without an express license agreement from NVIDIA Corporation is strictly
 * prohibited.
 *
 * TO THE MAXIMUM EXTENT PERMITTED BY APPLICABLE LAW, THIS SOFTWARE IS PROVIDED *AS IS*
 * AND NVIDIA AND ITS SUPPLIERS DISCLAIM ALL WARRANTIES, EITHER EXPRESS OR IMPLIED,
 * INCLUDING, BUT NOT LIMITED TO, IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
 * PARTICULAR PURPOSE.  IN NO EVENT SHALL NVIDIA OR ITS SUPPLIERS BE LIABLE FOR ANY
 * SPECIAL, INCIDENTAL, INDIRECT, OR CONSEQUENTIAL DAMAGES WHATSOEVER (INCLUDING, WITHOUT
 * LIMITATION, DAMAGES FOR LOSS OF BUSINESS PROFITS, BUSINESS INTERRUPTION, LOSS OF
 * BUSINESS INFORMATION, OR ANY OTHER PECUNIARY LOSS) ARISING OUT OF THE USE OF OR
 * INABILITY TO USE THIS SOFTWARE, EVEN IF NVIDIA HAS BEEN ADVISED OF THE POSSIBILITY OF
 * SUCH DAMAGES
 */

#include "ambocc.h"

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

__device__ __inline__ float gaussFilter(float dist, float zmin)
{

  //float scale = 0.5;                  //scale = 2*z_min*omegaShadeMax /omegaVMax
  float scale = zmin/2.0;
  float sample = dist/scale;
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

rtBuffer<float4, 2>              accum_buffer;
rtBuffer<float4, 2>              accum_buffer_occ;
//rtBuffer<float4, 2>              accum_buffer_occ_h;
rtBuffer<float3, 2>              world_loc;
rtBuffer<float, 2>               closest_intersection;
rtDeclareVariable(uint,           frame, , );
rtDeclareVariable(uint,           blur_occ, , );
rtDeclareVariable(uint,           err_vis, , );

rtDeclareVariable(uint,           normal_rpp, , );
rtDeclareVariable(uint,           brute_rpp, , );
rtDeclareVariable(uint,           show_progressive, , );
rtDeclareVariable(int2,          pixel_radius, , );

RT_PROGRAM void pinhole_camera() {

  size_t2 screen = output_buffer.size();

  float2 d = make_float2(launch_index) / make_float2(screen) * 2.f - 1.f;
  float3 ray_origin = eye;
  float3 ray_direction = normalize(d.x*U + d.y*V + W);

  if(frame == 0)
    closest_intersection[launch_index] = 10;


  float zmin = closest_intersection[launch_index];
  if (zmin < 0) {
    output_buffer[launch_index] = make_color(bg_color);
    return;
  }

  bool newInfo = false;
  if (frame < normal_rpp)
    newInfo = true;
  else if (zmin < 0.05 && frame < brute_rpp)
    newInfo = true;


  float4 acc_val = accum_buffer[launch_index];
  float4 acc_val_occ = accum_buffer_occ[launch_index];
  float3 cur_world_loc = make_float3(0.0);

  if (newInfo) {

    optix::Ray ray(ray_origin, ray_direction, radiance_ray_type, scene_epsilon );

    PerRayData_radiance prd;
    prd.importance = 1.f;
    prd.occlusion = make_float3(1.f);
    prd.depth = 0;
    prd.hit = false;

    rtTrace(top_object, ray, prd);

    if(!prd.hit) {
      closest_intersection[launch_index] = -1;
      output_buffer[launch_index] = make_color(bg_color);
      return;
    }

    cur_world_loc = prd.world_loc;
    world_loc[launch_index] = cur_world_loc;

    acc_val += (make_float4(prd.result,0.f) - acc_val)/(frame+1);
    acc_val_occ += (make_float4(prd.occlusion,prd.t_hit) - acc_val_occ)/(frame+1);
    accum_buffer[launch_index] = acc_val;
    accum_buffer_occ[launch_index] = acc_val_occ;

    zmin = min(zmin, prd.shadow_intersection);
    closest_intersection[launch_index] = zmin;
  } else {
    cur_world_loc = world_loc[launch_index];
  }

  float4 blurred_occ = make_float4(0.0);
  //int2 pixel_radius = make_int2(5,5);
  //pixel_radius = make_int2(10,10);
  float sumWeight = 0.0;

  //i guess just blur here for now... inefficient, but gets the point across
  if (frame > brute_rpp || (frame > 0 && show_progressive)) {
    int numBlurred = 0;

    for(int i=-pixel_radius.x; i < pixel_radius.x; i++) {
      for(int j=-pixel_radius.y; j < pixel_radius.y; j++) {
        if(launch_index.x + i > 0 && launch_index.y + j > 0) {
          if(launch_index.x + i < output_buffer.size().x && launch_index.y + j < output_buffer.size().y) {
            float4 target_occ = accum_buffer_occ[make_uint2(launch_index.x+i, launch_index.y+j)];
            //float distance = target_occ.w - prd.t_hit;
            float3 loca = cur_world_loc;
            float3 locb = world_loc[make_uint2(launch_index.x+i, launch_index.y+j)];
            float3 diff = loca-locb;
            float distance = sqrt(diff.x*diff.x + diff.y*diff.y + diff.z*diff.z);
            if(distance < 0)
              distance = -distance;
            if (zmin > 0.0) {
              float weight = gaussFilter(distance,zmin);
              blurred_occ += weight * target_occ;
              sumWeight += weight;
              if (weight > 0)
                numBlurred += 1;
            }
          }
        }
      }
    }
    if(sumWeight > 0)
      blurred_occ /= sumWeight;
    if(err_vis && numBlurred < 2)
      blurred_occ = make_float4(1,0,0,1);
    /*
    if(err_vis)
      //blurred_occ = make_float4(closest_intersection[launch_index]/10.0);
      blurred_occ = make_float4(prd.shadow_intersection/10.0);
      */
    if(!blur_occ)
      blurred_occ = acc_val_occ;
  } else if (frame == 0){
    closest_intersection[launch_index] = 100000.0f;
    blurred_occ = acc_val_occ;
  } else {
    blurred_occ = acc_val_occ;
  }


  output_buffer[launch_index] = make_color( make_float3(acc_val)*make_float3(blurred_occ.x, blurred_occ.y, blurred_occ.z));


}


//
// Returns solid color for miss rays
//
RT_PROGRAM void miss()
{
  prd_radiance.result = bg_color;
  prd_radiance.shadow_intersection = 100000.0f;
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
rtBuffer<BoxLight>        lights;
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

  float3 occlusion = make_float3(0.0);
  for(int i = 0; i < lights.size(); ++i) {
    BoxLight light = lights[i];

    uint2 seed = shadow_rng_seeds[launch_index];
    //seed.x = rot_seed(seed.x, frame);
    //seed.y = rot_seed(seed.y, frame);
    float2 sample = make_float2( rnd(seed.x), rnd(seed.y) );

    shadow_rng_seeds[launch_index] = seed;

    for(int i=0; i<1; ++i) {

      //From point, choose a random direction to sample in
      float3 U, V, W;
      float3 sampleDir; 
      createONB( ffnormal, U, V, W); //(is ffnormal the correct one to be using here?)
      sampleUnitHemisphere( sample, U, V, W, sampleDir );
      prd_radiance.shadow_intersection = 100;

      float nDl = dot( ffnormal, sampleDir );

      if( nDl > 0.0f ){
        //compute color
        color += Kd;

        float3 H = normalize(sampleDir - ray.direction);
        float nDh = dot( ffnormal, H );
        if (nDh > 0)
          color += Ks * pow(nDh, phong_exp);



        //cast ray and check for shadow
        PerRayData_shadow shadow_prd;
        shadow_prd.attenuation = make_float3(1.0f);
        optix::Ray shadow_ray ( hit_point, sampleDir, shadow_ray_type, scene_epsilon );
        rtTrace(top_shadower, shadow_ray, shadow_prd);
        occlusion += shadow_prd.attenuation;
        prd_radiance.shadow_intersection = shadow_prd.distance;
      }
    }


  }
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



  prd_radiance.occlusion = occlusion;
  prd_radiance.result = color;
}


//
// Set pixel to solid color upon failure
//
RT_PROGRAM void exception()
{
  output_buffer[launch_index] = make_color( bad_color );
}
