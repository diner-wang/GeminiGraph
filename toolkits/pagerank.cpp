#pragma clang diagnostic push
#pragma ide diagnostic ignored "cert-err34-c"
/*
Copyright (c) 2014-2015 Xiaowei Zhu, Tsinghua University

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
*/

#include <stdio.h>
#include <stdlib.h>

#include "core/graph.hpp"

#include <math.h>

const double d = (double)0.85;

void compute(Graph<Empty> * graph, int iterations) {
  double exec_time = 0;
  exec_time -= get_time();

  double * curr = graph->alloc_vertex_array<double>();
  double * next = graph->alloc_vertex_array<double>();
  VertexSubset * active = graph->alloc_vertex_subset();
  active->fill();

  double delta = graph->process_vertices<double>(
    [&](VertexId vtx){
      curr[vtx] = (double)1;
      if (graph->out_degree[vtx]>0) {
        curr[vtx] /= graph->out_degree[vtx];  // 每个边计算得到 1 / out_degree
      }
      return (double)1;
    },
    active
  );
  delta /= graph->vertices; // 计算一轮。得到变化的 vertex 占比

  for (int i_i=0;i_i<iterations;i_i++) {
    if (graph->partition_id==0) {
      printf("delta(%d)=%lf\n", i_i, delta);
    }
    graph->fill_vertex_array(next, (double)0);
    graph->process_edges<int,double>(
      [&](VertexId src){
        graph->emit(src, curr[src]);  // 写入的消息为 (src, curr[src])
      },
      [&](VertexId src, double msg, VertexAdjList<Empty> outgoing_adj){
        for (AdjUnit<Empty> * ptr=outgoing_adj.begin;ptr!=outgoing_adj.end;ptr++) {
          VertexId dst = ptr->neighbour;  // 遍历出边
          write_add(&next[dst], msg); // 将受到的消息加入 next[dst]
        }
        return 0;
      },
      [&](VertexId dst, VertexAdjList<Empty> incoming_adj) {
        double sum = 0;
        for (AdjUnit<Empty> * ptr=incoming_adj.begin;ptr!=incoming_adj.end;ptr++) {
          VertexId src = ptr->neighbour;
          sum += curr[src];
        } // 先 pull 再发送
        graph->emit(dst, sum);
      },
      [&](VertexId dst, double msg) {
        write_add(&next[dst], msg); // 将收到的消息求和
        return 0;
      },
      active
    );
    if (i_i==iterations-1) {
      delta = graph->process_vertices<double>(
        [&](VertexId vtx) {
          next[vtx] = 1 - d + d * next[vtx];
          return 0;
        },
        active
      );
    } else {
      delta = graph->process_vertices<double>(
        [&](VertexId vtx) {
          next[vtx] = 1 - d + d * next[vtx];
          if (graph->out_degree[vtx]>0) {
            next[vtx] /= graph->out_degree[vtx];
            return fabs(next[vtx] - curr[vtx]) * graph->out_degree[vtx];
          }
          return fabs(next[vtx] - curr[vtx]);
        },
        active
      );
    }
    delta /= graph->vertices; // 统计迭代步之间的差异
    std::swap(curr, next);
  }

  exec_time += get_time();
  if (graph->partition_id==0) {
    printf("exec_time=%lf(s)\n", exec_time);
  }

  double pr_sum = graph->process_vertices<double>(
    [&](VertexId vtx) {
      return curr[vtx];
    },
    active
  );
  if (graph->partition_id==0) {
    printf("pr_sum=%lf\n", pr_sum);
  }

  graph->gather_vertex_array(curr, 0);
  if (graph->partition_id==0) {
    VertexId max_v_i = 0;
    for (VertexId v_i=0;v_i<graph->vertices;v_i++) {
      if (curr[v_i] > curr[max_v_i]) max_v_i = v_i;
    }
    printf("pr[%u]=%lf\n", max_v_i, curr[max_v_i]);
  }

  graph->dealloc_vertex_array(curr);
  graph->dealloc_vertex_array(next);
  delete active;
}

int main(int argc, char ** argv) {
  MPI_Instance mpi(&argc, &argv);

  if (argc<4) {
    printf("pagerank [file] [vertices] [iterations]\n");
    exit(-1);
  }

  Graph<Empty> * graph;
  graph = new Graph<Empty>();
  graph->load_directed(argv[1], std::atoi(argv[2]));
  int iterations = std::atoi(argv[3]);

  compute(graph, iterations);
  for (int run=0;run<5;run++) {
    compute(graph, iterations);
  }

  delete graph;
  return 0;
}

#pragma clang diagnostic pop