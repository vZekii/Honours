OPENQASM 2.0;
include "qelib1.inc";
qreg q[18];
u3(pi/2,0,pi) q[9];
cx q[0],q[9];
u3(pi/2,0,pi) q[9];
u3(pi/2,0,pi) q[10];
cx q[1],q[10];
u3(pi/2,0,pi) q[10];
u3(pi/2,0,pi) q[11];
cx q[2],q[11];
u3(pi/2,0,pi) q[11];
u3(pi/2,0,pi) q[12];
cx q[3],q[12];
u3(pi/2,0,pi) q[12];
u3(pi/2,0,pi) q[13];
cx q[4],q[13];
u3(pi/2,0,pi) q[13];
u3(pi/2,0,pi) q[14];
cx q[5],q[14];
u3(pi/2,0,pi) q[14];
u3(pi/2,0,pi) q[15];
cx q[6],q[15];
u3(pi/2,0,pi) q[15];
u3(pi/2,0,pi) q[16];
cx q[7],q[16];
u3(pi/2,0,pi) q[16];
u3(pi/2,0,pi) q[17];
cx q[8],q[17];
u3(pi/2,0,pi) q[17];