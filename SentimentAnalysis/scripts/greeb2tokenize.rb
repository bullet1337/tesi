#!/usr/bin/ruby
require 'greeb'

if __FILE__ == $0
	text = ""
	ARGF.each do |line|
		text.concat(line)
	end
	tokens = Greeb::Tokenizer.tokenize(text)
	tokens.each do |x|
		if x.type.to_s != "space" and x.type.to_s != "break"
			puts "%s\t%d\t%d\t%s\n" % [text.slice(x.from..x.to - 1), x.from, x.to, x.type]
		end
	end
end
