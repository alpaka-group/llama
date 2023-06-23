This example is a LLAMA version from the IO benchmark found here:
https://github.com/jblomer/iotools/blob/master/lhcb.cxx

The lhcb analysis example requires an input file, which can be downloaded here:
https://root.cern/files/RNTuple/.
The file is typically called B2HHH~zstd.ntuple, so you can run:
`wget https://root.cern/files/RNTuple/B2HHH~zstd.ntuple`
There are also artificially enlarged datasets available,
called `B2HHHX10~zstd.ntuple` (10x the size) and `B2HHHX25~zstd.ntuple` (25x the size),
for benchmarking on larger systems.

If you get an error due to a version incompatibility of the file,
you can create an RNTuple file yourself from a TTree file as a workaround
(see also [this GitHub issue](https://github.com/jblomer/iotools/issues/9)).
This way, you can also produce artifically enlarged datasets yourself:

Download the TTree file:
```bash
wget https://root.cern/files/RNTuple/treeref/B2HHH~zstd.root
```

Convert to a (larger) RNTuple file by running inside the ROOT interpreter:
```c++
TChain chain;
for (int i = 0; i < 25; i++) chain.Add("B2HHH~zstd.root?#DecayTree");
auto imp = ROOT::Experimental::RNTupleImporter::Create(&chain, "B2HHH~zstd~25x.ntuple");
imp->SetNTupleName("DecayTree");
imp->Import();
```
