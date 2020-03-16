# importing converter
using Weave

old_dir = pwd()
cd(@__DIR__)
files = readdir()

# for each file check to see if it is a Jupyter notebook. If so make a converted version
for file in files

    # prevents short file names from causing errors
    if length(file) < 6
		continue
	end
	if ".ipynb"==file[end-5:end]
    	println(file)
        convert_doc(file, file[1:end-6] * ".jmd")
	end

end

# change back to the upper directory
cd(old_dir)
